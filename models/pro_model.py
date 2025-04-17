# pro_model.py（融合预测 PRO 模型）

import os
import joblib
import numpy as np
import pandas as pd

# 获取当前文件目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_weights", "pro_model")

# 加载模型和辅助文件（权重组合：rf=0.3, lr=0.3, xgb=0.4）
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def predict_model_pro(input_df: pd.DataFrame) -> pd.DataFrame:
    X = input_df[feature_cols]

    rf_prob = rf_model.predict_proba(X)
    lr_prob = lr_model.predict_proba(X)
    xgb_prob = xgb_model.predict_proba(X)

    # 加权融合
    fused_prob = 0.3 * rf_prob + 0.3 * lr_prob + 0.4 * xgb_prob

    # 预测结果与信心
    top1 = fused_prob.argmax(axis=1)
    top1_label = label_encoder.inverse_transform(top1)
    gap = np.ptp(fused_prob, axis=1)  # 极差作为融合信心分

    result_df = input_df.copy()
    result_df["最终预测结果"] = top1_label
    result_df["average_gap"] = gap
    return result_df
