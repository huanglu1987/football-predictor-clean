# app.py
# ------------------------------------------------------------------
# Football Match Predictor — 全部使用相对路径
# ------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ↓↓↓ 现有模型
from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble
# ↓↓↓ META 融合模型
from models.predict_model_meta import predict_model_meta

from similarity_matcher import SimilarityMatcher

# ────────── 根目录 & 路径配置 ──────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
MODELS_DIR= BASE_DIR / "models"
HIST_PATH = DATA_DIR / "prediction_results (43).xlsx"   # 相对 data 文件夹

# ────────── 页面配置 ──────────
st.set_page_config(page_title="Football Match Predictor", layout="wide")
st.title("⚽ Football Match Predictor")

# ────────── 常量定义 ──────────
company_order = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
outcomes      = ["主胜", "平局", "客胜"]
feature_cols  = [f"{cmp}_{oc}" for cmp in company_order for oc in outcomes]

# ────────── Session State 初始化 ──────────
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = pd.DataFrame(columns=feature_cols)
if "current_odds" not in st.session_state:
    st.session_state.current_odds = []
if "input_odds_line" not in st.session_state:
    st.session_state.input_odds_line = ""
if "matcher" not in st.session_state:
    # SimilarityMatcher 接受 Path 或 str 皆可
    st.session_state.matcher = SimilarityMatcher(str(HIST_PATH))

# ────────── 辅助函数 ──────────
def handle_input_odds():
    """处理一行“主胜 平局 客胜”输入"""
    line = st.session_state.input_odds_line.strip()
    parts = line.split()
    if len(parts) == 3:
        try:
            odds = list(map(float, parts))
            if len(st.session_state.current_odds) < 5:
                st.session_state.current_odds.append(odds)
                idx = len(st.session_state.current_odds) - 1
                st.success(f"✅ 已添加：{company_order[idx]}")
            else:
                st.warning("已录入 5 家公司赔率，如需修改请撤回。")
        except ValueError:
            st.error("❌ 赔率必须为数字")
    elif line:
        st.error("❌ 格式应为：主胜 平局 客胜")
    st.session_state.input_odds_line = ""


def compute_upset_score(row):
    """基于原始赔率计算冷门原始分 = max(avg) - min(avg)"""
    avg_home = np.mean([row[f"{c}_主胜"] for c in company_order])
    avg_draw = np.mean([row[f"{c}_平局"] for c in company_order])
    avg_away = np.mean([row[f"{c}_客胜"] for c in company_order])
    return float(max(avg_home, avg_draw, avg_away) - min(avg_home, avg_draw, avg_away))


def generate_recommendation_score(row):
    """计算融合信心并标记模型一致"""
    row["融合信心"] = round(0.6 * row["PRO融合模型_gap"] + 0.4 * row["PRO_gap"], 4)
    row["模型一致"] = (row["PRO_最终预测结果"] == row["PRO融合模型预测结果"])
    return row


def classify_recommendation(sc):
    if sc >= 0.80:
        return "⭐⭐⭐⭐ 强烈推荐"
    elif sc >= 0.60:
        return "⭐⭐⭐ 可博一试"
    elif sc >= 0.40:
        return "⭐⭐ 潜力场次"
    elif sc >= 0.20:
        return "⭐ 一般场次"
    else:
        return "❌ 不建议投注"

# ────────── 用户输入 ──────────
mode = st.radio(
    "请选择比赛数据输入方式",
    ["📁 上传文件", "🖊️ 手动录入（逐公司一行）"]
)

if mode == "📁 上传文件":
    uploaded = st.file_uploader("上传包含 15 列赔率的文件 (Excel 或 CSV)", type=["xlsx", "csv"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded)
        else:
            df_upload = pd.read_excel(uploaded)
        df_upload = df_upload[feature_cols]
        st.session_state.uploaded_df = df_upload
        st.success(f"✅ 已读取 {len(df_upload)} 场比赛")
        st.dataframe(df_upload)
else:
    st.markdown(f"🎯 录入顺序：**{' → '.join(company_order)}**")
    st.text_input(
        "✏️ 粘贴赔率（主胜 平局 客胜）",
        key="input_odds_line",
        on_change=handle_input_odds
    )
    if st.session_state.current_odds:
        tmp = pd.DataFrame(
            st.session_state.current_odds,
            columns=outcomes,
            index=company_order[:len(st.session_state.current_odds)]
        )
        st.table(tmp)
        if st.button("🔙 撤回上一家公司"):
            st.session_state.current_odds.pop()
            st.warning("已撤回上一家公司赔率")
    if len(st.session_state.current_odds) == 5:
        row = {
            f"{company_order[i]}_{oc}": val
            for i, odds in enumerate(st.session_state.current_odds)
            for oc, val in zip(outcomes, odds)
        }
        df_manual = pd.DataFrame([row], columns=feature_cols)
        st.session_state.uploaded_df = pd.concat(
            [st.session_state.uploaded_df, df_manual],
            ignore_index=True
        )
        st.session_state.current_odds = []
        st.success("🎉 已添加 1 场比赛")
        st.dataframe(st.session_state.uploaded_df)

# ────────── 预测与展示 ──────────
# A. PRO + 融合 模型
if st.button("🧠 同时运行 PRO + 融合模型预测"):
    df_in = st.session_state.uploaded_df
    if df_in.empty:
        st.error("❌ 请先上传或录入至少一场比赛的数据。")
    else:
        df_filled = df_in.copy()
        df_filled[feature_cols] = df_filled[feature_cols].fillna(df_filled[feature_cols].mean())

        # PRO
        df_pro = predict_model_pro(df_filled)
        pro_preds = df_pro["最终预测结果"].values
        pro_gaps  = df_pro["average_gap"].values

        # PRO 融合
        df_ens = predict_model_pro_ensemble(df_filled)
        fusion_preds = df_ens["PRO融合模型预测结果"].values
        fusion_gaps  = df_ens["PRO融合模型_gap"].values

        prob_cols = [f"P({oc})" for oc in outcomes]
        probs = df_ens[prob_cols].values if set(prob_cols).issubset(df_ens.columns) else np.zeros((len(df_filled), 3))

        result_df = df_filled.reset_index(drop=True)
        result_df["比赛序号"]             = np.arange(1, len(df_filled)+1)
        result_df["PRO_最终预测结果"]    = pro_preds
        result_df["PRO_gap"]             = pro_gaps
        result_df["PRO融合模型预测结果"]  = fusion_preds
        result_df["PRO融合模型_gap"]      = fusion_gaps
        result_df["P(主胜)"]             = probs[:,0]
        result_df["P(平局)"]             = probs[:,1]
        result_df["P(客胜)"]             = probs[:,2]

        result_df = result_df.apply(generate_recommendation_score, axis=1)
        result_df["冷门原始分"]     = result_df.apply(compute_upset_score, axis=1)
        result_df["冷门标准化得分"] = ((result_df["冷门原始分"]-2.16)/1.47).round(2)
        result_df["推荐总分"]       = (0.7*result_df["融合信心"]-0.3*result_df["冷门标准化得分"]).round(2)
        result_df["推荐等级"]       = result_df["推荐总分"].apply(classify_recommendation)

        st.subheader("📊 PRO + 融合模型预测结果")
        st.dataframe(
            result_df[[
                "比赛序号","PRO_最终预测结果","PRO_gap",
                "PRO融合模型预测结果","PRO融合模型_gap",
                "融合信心","推荐总分"
            ]].style.format({
                "PRO_gap":"{:.4f}","PRO融合模型_gap":"{:.4f}",
                "融合信心":"{:.4f}","推荐总分":"{:.2f}"
            })
        )

        st.markdown("## 🔍 相似历史比赛（Top-5）")
        matcher = st.session_state.matcher
        for _, row in result_df.iterrows():
            query = {
                "PRO_gap": row["PRO_gap"],
                "PRO融合模型_gap": row["PRO融合模型_gap"],
                "融合信心": row["融合信心"],
                "推荐总分": row["推荐总分"],
                "pair": f"{row['PRO_最终预测结果']}-{row['PRO融合模型预测结果']}"
            }
            top5 = matcher.query(query, k=5)
            with st.expander(f"比赛 {int(row['比赛序号'])} → 预测 {row['PRO_最终预测结果']}"):
                st.dataframe(
                    top5[[
                        "比赛序号","比赛结果",
                        "PRO_最终预测结果","PRO融合模型预测结果",
                        "PRO_gap","PRO融合模型_gap",
                        "融合信心","推荐总分","_distance"
                    ]].style.format({
                        "PRO_gap":"{:.4f}","PRO融合模型_gap":"{:.4f}",
                        "融合信心":"{:.4f}","推荐总分":"{:.2f}","_distance":"{:.2f}"
                    })
                )

# B. META 融合模型
if st.button("🧠 运行 META 融合模型预测"):
    df_in = st.session_state.uploaded_df
    if df_in.empty:
        st.error("❌ 请先上传或录入至少一场比赛的数据。")
    else:
        df_filled = df_in.copy()
        df_filled[feature_cols] = df_filled[feature_cols].fillna(df_filled[feature_cols].mean())

        # 补齐 PRO 列
        df_pro = predict_model_pro(df_filled)
        df_filled["PRO_最终预测结果"]    = df_pro["最终预测结果"]
        df_filled["PRO_gap"]             = df_pro["average_gap"]
        df_filled["PRO融合模型预测结果"]  = df_filled.get("PRO融合模型预测结果", df_filled["PRO_最终预测结果"])
        df_filled["PRO融合模型_gap"]      = df_filled.get("PRO融合模型_gap", df_filled["PRO_gap"])

        # 补齐其他必需列
        for col in ["模型一致","融合信心","P(主胜)","P(平局)","P(客胜)","推荐总分","冷门标准化得分"]:
            if col not in df_filled:
                df_filled[col] = 0

        df_meta = predict_model_meta(df_filled)
        df_meta.insert(0, "比赛序号", np.arange(1, len(df_meta)+1))

        st.subheader("📊 META 融合模型预测结果（标签 + 置信度）")
        st.dataframe(df_meta.style.format({"预测置信度":"{:.4f}"}))
