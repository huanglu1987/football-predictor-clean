import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import requests
import sys

# —— 自动定位本文件目录下的模型文件 —— 
MODEL_PATH = Path(__file__).resolve().parent / "rf_meta_calibrated.pkl"

# —— Release 里模型的下载地址 —— 
# 根据你的 GitHub 页面自动填写
MODEL_URL = (
    "https://github.com/huanglu1987/football-predictor-clean/"
    "releases/download/v1.0.0-meta/rf_meta_calibrated.pkl"
)

def download_model():
    """如果本地模型文件不存在，就从 Release 下载"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 本地未找到模型，尝试从 Release 下载：{MODEL_URL}", file=sys.stderr)
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 模型下载完成并保存到：{MODEL_PATH}", file=sys.stderr)

# 在首次 import 时，若模型文件缺失即触发下载
if not MODEL_PATH.exists():
    try:
        download_model()
    except Exception as e:
        raise RuntimeError(f"无法下载模型，请检查 MODEL_URL：{e}")

# 最后加载模型
rf_model = joblib.load(MODEL_PATH)


def make_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """
    从原始赔率 + PRO/PRO融合输出构建元模型特征矩阵。
    """
    companies = ['Bet365', '立博', 'Interwetten', 'Pinnacle', 'William Hill']
    outcomes  = ['主胜', '平局', '客胜']
    odds_cols = [f"{c}_{o}" for c in companies for o in outcomes]

    df_feat = df.copy()

    # 衍生 gap & std
    for o in outcomes:
        cols = [f"{c}_{o}" for c in companies]
        df_feat[f"{o}_gap"] = df_feat[cols].max(axis=1) - df_feat[cols].min(axis=1)
        df_feat[f"{o}_std"] = df_feat[cols].std(axis=1)

    # 确保必要列存在
    defaults = {
        'PRO_最终预测结果':   '平局',
        'PRO融合模型预测结果': '平局',
        'PRO_gap':            0.0,
        'PRO融合模型_gap':    0.0,
        '融合信心':            0.0,
        '模型一致':            0,
        'P(主胜)':             0.0,
        'P(平局)':             0.0,
        'P(客胜)':             0.0,
        '推荐总分':            0.0,
        '冷门标准化得分':      0.0
    }
    for col, default in defaults.items():
        if col not in df_feat:
            df_feat[col] = default

    # 类别编码
    mapping = {'主胜': 0, '平局': 1, '客胜': 2}
    df_feat['PRO预测']     = df_feat['PRO_最终预测结果'].map(mapping).fillna(1).astype(int)
    df_feat['PRO融合预测'] = df_feat['PRO融合模型预测结果'].map(mapping).fillna(1).astype(int)
    df_feat['模型一致']    = df_feat['模型一致'].astype(int, errors='ignore').fillna(0).astype(int)

    # 组合特征
    feat_cols = (
        odds_cols +
        [f"{o}_gap" for o in outcomes] +
        [f"{o}_std" for o in outcomes] +
        ['PRO_gap', 'PRO融合模型_gap', '融合信心',
         'P(主胜)', 'P(平局)', 'P(客胜)',
         'PRO预测', 'PRO融合预测', '模型一致',
         '推荐总分', '冷门标准化得分']
    )
    feats = df_feat[feat_cols].fillna(0)
    return feats


def predict_model_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入包含原始赔率与 PRO/PRO融合输出的 DataFrame，
    返回 META 模型的 {预测结果, 预测置信度} 两列表。
    """
    # 构造特征
    feats_df = make_features_fixed(df)

    # 对齐训练时的特征顺序
    train_cols = getattr(rf_model, "feature_names_in_", None)
    if train_cols is not None:
        # 补齐缺失列、丢弃多余列
        for col in train_cols:
            if col not in feats_df.columns:
                feats_df[col] = 0
        feats_df = feats_df[list(train_cols)]

    # 预测
    probs = rf_model.predict_proba(feats_df.values)
    preds = probs.argmax(axis=1)

    # 整理输出
    mapping = {0: "主胜", 1: "平局", 2: "客胜"}
    result = pd.DataFrame({
        "预测结果":     [mapping[p] for p in preds],
        "预测置信度":  probs.max(axis=1).round(4)
    })
    return result
