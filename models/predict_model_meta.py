import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import requests
import sys

# —— 相对定位：本脚本同目录下的模型文件 —— 
MODEL_PATH = Path(__file__).resolve().parent / "rf_meta_calibrated.pkl"

# —— Release Asset 下载地址 —— 
MODEL_URL = (
    "https://github.com/huanglu1987/football-predictor-clean/"
    "releases/download/v1.0.0-meta/rf_meta_calibrated.pkl"
)

def download_model():
    """如果本地缺少模型文件，则从 Release 自动下载"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 模型未找到，开始下载：{MODEL_URL}", file=sys.stderr)
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ 模型下载完毕：{MODEL_PATH}", file=sys.stderr)

# 在 import 时确保模型文件存在
if not MODEL_PATH.exists():
    try:
        download_model()
    except Exception as e:
        raise RuntimeError(f"无法下载模型，请检查 MODEL_URL 和网络：{e}")

# 加载模型
rf_model = joblib.load(MODEL_PATH)


def make_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：构造与训练时相同的特征矩阵
    """
    companies = ['Bet365','立博','Interwetten','Pinnacle','William Hill']
    outcomes  = ['主胜','平局','客胜']
    # 原始赔率列
    odds_cols = [f"{c}_{o}" for c in companies for o in outcomes]

    df_feat = df.copy()
    # 衍生 gap & std
    for o in outcomes:
        cols = [f"{c}_{o}" for c in companies]
        df_feat[f"{o}_gap"] = df_feat[cols].max(axis=1) - df_feat[cols].min(axis=1)
        df_feat[f"{o}_std"] = df_feat[cols].std(axis=1)
    # 默认值填充
    defaults = {
        'PRO_gap': 0.0,
        'PRO融合模型_gap': 0.0,
        '融合信心': 0.0,
        '模型一致': 0,
        'P(主胜)':0.0,'P(平局)':0.0,'P(客胜)':0.0,
    }
    for k, v in defaults.items():
        if k not in df_feat:
            df_feat[k] = v
    # 编码预测结果字段
    mapping = {'主胜':0,'平局':1,'客胜':2}
    df_feat['PRO预测']     = df_feat.get('PRO_最终预测结果','平局').map(mapping).fillna(1).astype(int)
    df_feat['PRO融合预测'] = df_feat.get('PRO融合模型预测结果','平局').map(mapping).fillna(1).astype(int)
    df_feat['模型一致']    = df_feat['模型一致'].astype(int, errors='ignore').fillna(0).astype(int)

    # 对齐训练时的列顺序
    train_cols = list(rf_model.feature_names_in_)
    feats = df_feat[train_cols].fillna(0)
    return feats


def predict_model_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入包含赔率及 PRO/融合模型输出的 DataFrame，
    返回 Top-1 和 Top-2 预测及其置信度
    """
    feats_df = make_features_fixed(df)
    # 预测概率
    probs = rf_model.predict_proba(feats_df.values)
    # 每行取前两高索引
    idx_sorted = np.argsort(probs, axis=1)[:, ::-1]
    top1_idx = idx_sorted[:, 0]
    top2_idx = idx_sorted[:, 1]

    # 标签映射
    label_map = {0:'主胜',1:'平局',2:'客胜'}
    pred1 = [label_map[i] for i in top1_idx]
    pred2 = [label_map[i] for i in top2_idx]
    conf1 = probs[np.arange(len(probs)), top1_idx].round(4)
    conf2 = probs[np.arange(len(probs)), top2_idx].round(4)

    # 返回四列：预测1/置信度1，预测2/置信度2
    return pd.DataFrame({
        '预测1':    pred1,
        '置信度1': conf1,
        '预测2':    pred2,
        '置信度2': conf2
    })
