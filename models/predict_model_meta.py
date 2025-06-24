import pandas as pd
import joblib
import numpy as np

# 载入模型（请确保路径正确）
MODEL_PATH = "/Users/huanglu/Desktop/football_predictor/models/rf_meta_calibrated.pkl"
rf_model = joblib.load(MODEL_PATH)

def make_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    odds_cols = [col for col in df.columns if any(b in col for b in ['Bet365', '立博', 'Interwetten', 'Pinnacle', 'William Hill'])]
    df_feat = df.copy()
    
    # 衍生结构差值与标准差
    df_feat['主胜_gap'] = df_feat[[col for col in odds_cols if '主胜' in col]].max(axis=1) - df_feat[[col for col in odds_cols if '主胜' in col]].min(axis=1)
    df_feat['平局_gap'] = df_feat[[col for col in odds_cols if '平局' in col]].max(axis=1) - df_feat[[col for col in odds_cols if '平局' in col]].min(axis=1)
    df_feat['客胜_gap'] = df_feat[[col for col in odds_cols if '客胜' in col]].max(axis=1) - df_feat[[col for col in odds_cols if '客胜' in col]].min(axis=1)

    df_feat['主胜_std'] = df_feat[[col for col in odds_cols if '主胜' in col]].std(axis=1)
    df_feat['平局_std'] = df_feat[[col for col in odds_cols if '平局' in col]].std(axis=1)
    df_feat['客胜_std'] = df_feat[[col for col in odds_cols if '客胜' in col]].std(axis=1)

    df_feat['PRO预测'] = df_feat['PRO_最终预测结果'].map({'主胜': 0, '平局': 1, '客胜': 2})
    df_feat['PRO融合预测'] = df_feat['PRO融合模型预测结果'].map({'主胜': 0, '平局': 1, '客胜': 2})
    df_feat['模型一致'] = df_feat['模型一致'].astype(int)

    feats = pd.concat([
        df_feat[odds_cols],
        df_feat[[
            '主胜_gap', '平局_gap', '客胜_gap',
            '主胜_std', '平局_std', '客胜_std',
            'PRO_gap', '融合信心',
            'P(主胜)', 'P(平局)', 'P(客胜)',
            'PRO预测', 'PRO融合预测', '模型一致',
            '推荐总分', '冷门标准化得分'
        ]]
    ], axis=1).fillna(0)

    return feats

def predict_model_meta(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 特征工程
    feats_df = make_features_fixed(df)

    # 2) 对齐训练时的特征
    train_cols = getattr(rf_model, "feature_names_in_", None)
    if train_cols is not None:
        for col in train_cols:
            if col not in feats_df.columns:
                feats_df[col] = 0
        feats_df = feats_df[list(train_cols)]

    # 3) 预测概率矩阵 (n_samples, 3)
    probs = rf_model.predict_proba(feats_df.values)
    # 对每行取两大
    top2_idx = np.argsort(probs, axis=1)[:, ::-1][:, :2]  # shape (n,2)
    top1_idx = top2_idx[:, 0]
    top2_idx = top2_idx[:, 1]

    mapping = {0: "主胜", 1: "平局", 2: "客胜"}
    top1_label = [mapping[i] for i in top1_idx]
    top2_label = [mapping[i] for i in top2_idx]
    top1_prob  = probs[np.arange(len(probs)), top1_idx].round(4)
    top2_prob  = probs[np.arange(len(probs)), top2_idx].round(4)

    # 4) 整理输出
    return pd.DataFrame({
        "top1_label":  top1_label,
        "top1_prob":   top1_prob,
        "top2_label":  top2_label,
        "top2_prob":   top2_prob
    })
