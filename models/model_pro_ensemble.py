import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "/football_predictor_clean/model_weights/final_model_with_structure_features"

# 加载模型及工具
ensemble_model = joblib.load(f"{MODEL_DIR}/final_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

def predict_model_pro_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    # 确保结构特征存在（必要时动态添加）
    if "赔率差异指数" not in df.columns or "公司间一致性指数" not in df.columns:
        company_order = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
        all_odds_cols = [f"{c}_{r}" for c in company_order for r in ["主胜", "平局", "客胜"]]
        df["赔率差异指数"] = df[all_odds_cols].std(axis=1)
        df["公司间一致性指数"] = (
            df[[f"{c}_主胜" for c in company_order]].std(axis=1) +
            df[[f"{c}_平局" for c in company_order]].std(axis=1) +
            df[[f"{c}_客胜" for c in company_order]].std(axis=1)
        )

    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    preds = ensemble_model.predict(X_scaled)
    decoded_preds = label_encoder.inverse_transform(preds)

    proba = ensemble_model.predict_proba(X_scaled)
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    gap = (sorted_proba[:, 0] - sorted_proba[:, 1]).round(4)

    return pd.DataFrame({
    "PRO融合模型预测结果": decoded_preds,
    "PRO融合模型_gap": gap,
    "P(主胜)": np.round(proba[:, label_encoder.transform(["主胜"])[0]], 4),
    "P(平局)": np.round(proba[:, label_encoder.transform(["平局"])[0]], 4),
    "P(客胜)": np.round(proba[:, label_encoder.transform(["客胜"])[0]], 4),
    "max_diff12": (sorted_proba[:, 0] - sorted_proba[:, 1]).round(4),
    "diff23": (sorted_proba[:, 1] - sorted_proba[:, 2]).round(4),
    })
