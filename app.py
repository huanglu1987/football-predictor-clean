import streamlit as st
import pandas as pd
import numpy as np
from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble

# 设置页面配置与标题
st.set_page_config(page_title="Football Match Predictor", layout="wide")
st.title("⚽ Football Match Predictor")

company_order = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]

if "current_odds" not in st.session_state:
    st.session_state.current_odds = []
if "input_odds_line" not in st.session_state:
    st.session_state.input_odds_line = ""
if "matches" not in st.session_state:
    st.session_state.matches = []

# 1. 手动输入赔率处理函数
def handle_input_odds():
    input_line = st.session_state.input_odds_line.strip()
    parts = input_line.split()
    if len(parts) == 3:
        try:
            odds = list(map(float, parts))
            if len(st.session_state.current_odds) < 5:
                st.session_state.current_odds.append(odds)
                current_index = len(st.session_state.current_odds) - 1
                st.success(f"✅ 已添加：{company_order[current_index]}")
            else:
                st.warning("当前比赛已录入5家公司，请先确认添加比赛或撤回一项。")
        except ValueError:
            st.error("❌ 赔率应为数字")
    elif input_line:
        st.error("❌ 格式错误，应为：主胜 平局 客胜")
    st.session_state.input_odds_line = ""

# 1. 手动输入赔率处理函数
def handle_input_odds():
    input_line = st.session_state.input_odds_line.strip()
    parts = input_line.split()
    if len(parts) == 3:
        try:
            odds = list(map(float, parts))
            if len(st.session_state.current_odds) < 5:
                st.session_state.current_odds.append(odds)
                current_index = len(st.session_state.current_odds) - 1
                st.success(f"✅ 已添加：{company_order[current_index]}")
            else:
                st.warning("当前比赛已录入5家公司，请先确认添加比赛或撤回一项。")
        except ValueError:
            st.error("❌ 赔率应为数字")
    elif input_line:
        st.error("❌ 格式错误，应为：主胜 平局 客胜")
    st.session_state.input_odds_line = ""

# 2. 辅助函数
def calculate_consistency(row):
    std_home = np.std([row[f"{c}_主胜"] for c in company_order])
    std_draw = np.std([row[f"{c}_平局"] for c in company_order])
    std_away = np.std([row[f"{c}_客胜"] for c in company_order])
    return std_home + std_draw + std_away

def assign_consistency_level(index):
    if index < 0.3:
        return "高一致"
    elif index < 0.6:
        return "中一致"
    else:
        return "低一致"

def compute_upset_score(row):
    avg_home = np.mean([row[f"{c}_主胜"] for c in company_order])
    avg_draw = np.mean([row[f"{c}_平局"] for c in company_order])
    avg_away = np.mean([row[f"{c}_客胜"] for c in company_order])
    return max(avg_home, avg_draw, avg_away) - min(avg_home, avg_draw, avg_away)

def generate_recommendation_score(row):
    row["融合信心"] = round(0.6 * row["PRO融合模型_gap"] + 0.4 * row["PRO_gap"], 4)
    row["模型一致"] = row["PRO_最终预测结果"] == row["PRO融合模型预测结果"]
    return row

def classify_recommendation(score):
    if score >= 0.80:
        return "⭐⭐⭐⭐ 强烈推荐"
    elif score >= 0.60:
        return "⭐⭐⭐ 可博一试"
    elif score >= 0.40:
        return "⭐⭐ 潜力场次"
    elif score >= 0.20:
        return "⭐ 一般场次"
    else:
        return "❌ 不建议投注"

def generate_ai_advice(row):
    if row["模型一致"] and row["PRO_gap"] >= 0.4 and row["PRO融合模型_gap"] >= 0.3:
        return f"🚀 超强一致推荐：{row['PRO融合模型预测结果']}"
    elif row["PRO_gap"] >= 0.4 and row["推荐总分"] > 0.15:
        return f"✅ AI推荐投注（PRO）：{row['PRO_最终预测结果']}"
    elif row["PRO_gap"] < 0.4 and row["PRO融合模型_gap"] > 0.2 and row["推荐总分"] > 0.4:
        return f"✅ AI推荐投注（融合）：{row['PRO融合模型预测结果']}"
    else:
        return "🧠 人工判定"

# 生成双预测字段（满足任一组合条件）
def generate_dual_prediction(row):
    cond_A = (
        row["PRO融合模型_gap"] > 0.2 and
        row["推荐总分"] < 0.4 and
        row["max_diff12"] < 0.1
    )
    cond_B = (
        row["PRO融合模型_gap"] <= 0.2 and
        row["推荐总分"] < 0.5 and
        row["max_diff12"] < 0.1
    )
    if cond_A or cond_B:
        probs = {
            "主胜": row["P(主胜)"],
            "平局": row["P(平局)"],
            "客胜": row["P(客胜)"]
        }
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return f"{sorted_probs[0][0]}｜{sorted_probs[1][0]}"
    else:
        return row["PRO融合模型预测结果"]

# ----------------------------
# 4. 选择比赛数据输入方式
# ----------------------------
mode = st.radio("请选择比赛数据输入方式", ["📁 上传Excel", "🖊️ 手动录入（逐公司一行）"])
df = None

if mode == "📁 上传Excel":
    uploaded_file = st.file_uploader("上传比赛Excel文件（.xlsx）", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        # 让索引从1开始
        df.index = df.index + 1
        df.index.name = "比赛序号"
        st.success("✅ 文件读取成功")
        st.dataframe(df.head())

elif mode == "🖊️ 手动录入（逐公司一行）":
    st.markdown(f"🎯 公司录入顺序固定为：**{' → '.join(company_order)}**")
    st.text_input("✏️ 粘贴赔率（主胜 平局 客胜）", key="input_odds_line", on_change=handle_input_odds)

    if st.button("🔙 撤回上一家公司"):
        if st.session_state.current_odds:
            st.session_state.current_odds.pop()
            st.warning("已撤回上一家公司赔率")

    if st.session_state.current_odds:
        display_df = pd.DataFrame(
            st.session_state.current_odds,
            columns=["主胜", "平局", "客胜"],
            index=company_order[:len(st.session_state.current_odds)]
        )
        st.markdown("#### 当前公司赔率录入表")
        st.dataframe(display_df.style.format("{:.2f}"))

    if len(st.session_state.current_odds) == 5 and st.button("✅ 确认添加该场比赛"):
        match_row = {}
        for idx, odds in enumerate(st.session_state.current_odds):
            cname = company_order[idx]
            match_row[f"{cname}_主胜"] = odds[0]
            match_row[f"{cname}_平局"] = odds[1]
            match_row[f"{cname}_客胜"] = odds[2]
        st.session_state.matches.append(match_row)
        st.session_state.current_odds = []
        st.success("🎉 成功添加一场比赛！")

    if st.session_state.matches:
        df = pd.DataFrame(st.session_state.matches)
        df.index = df.index + 1
        df.index.name = "比赛序号"
        st.dataframe(df)


# 5. 进行模型预测 + 展示结果
if st.button("🧠 同时运行 PRO + 融合模型预测"):
    if df is not None and not df.empty:
        result_pro = predict_model_pro(df)
        result_ens = predict_model_pro_ensemble(df)

        result_df = df.copy()
        result_df["PRO_最终预测结果"] = result_pro["最终预测结果"]
        result_df["PRO_gap"] = result_pro["average_gap"]
        result_df["PRO融合模型预测结果"] = result_ens["PRO融合模型预测结果"].values
        result_df["PRO融合模型_gap"] = result_ens["PRO融合模型_gap"].values
        result_df["P(主胜)"] = result_ens["P(主胜)"].values
        result_df["P(平局)"] = result_ens["P(平局)"].values
        result_df["P(客胜)"] = result_ens["P(客胜)"].values
        result_df["diff23"] = result_ens["diff23"].values
        result_df["max_diff12"] = result_ens["max_diff12"].values

        result_df["一致性指数"] = result_df.apply(calculate_consistency, axis=1)
        result_df["一致性等级"] = result_df["一致性指数"].apply(assign_consistency_level)
        result_df["冷门原始分"] = result_df.apply(compute_upset_score, axis=1)
        result_df["冷门标准化得分"] = ((result_df["冷门原始分"] - 2.16) / 1.47).round(2)
        result_df = result_df.apply(generate_recommendation_score, axis=1)
        result_df["推荐总分"] = (0.7 * result_df["融合信心"] - 0.3 * result_df["冷门标准化得分"]).round(2)
        result_df["推荐等级"] = result_df["推荐总分"].apply(classify_recommendation)
        result_df["AI建议投注"] = result_df.apply(generate_ai_advice, axis=1)
        result_df["双预测推荐"] = result_df.apply(generate_dual_prediction, axis=1)

        # 添加不建议投注判断逻辑
        def is_not_recommended(row):
            conditions_met = 0
            if row["PRO融合模型_gap"] < 0.1:
                conditions_met += 1
            if row["推荐总分"] < 0.05:
                conditions_met += 1
            if row["diff23"] < 0.1:
                conditions_met += 1
            if row["max_diff12"] < 0.1:
                conditions_met += 1
            return conditions_met >= 3
        
        # 应用判断逻辑
        result_df["不建议投注标记"] = result_df.apply(is_not_recommended, axis=1)
        result_df["不建议投注"] = result_df["不建议投注标记"].map({True: "❌ 不建议", False: ""})

        # 确保按比赛序号排序输出
        if "比赛序号" in result_df.columns:
            result_df_sorted = result_df.set_index("比赛序号").sort_index()
        else:
            result_df_sorted = result_df.sort_index()

        st.subheader("📊 模型预测结果")
        display_df = result_df_sorted[[
            "PRO_最终预测结果", "PRO_gap",
            "PRO融合模型预测结果", "PRO融合模型_gap",
            "P(主胜)", "P(平局)", "P(客胜)",
             "推荐总分","双预测推荐","AI建议投注", "不建议投注"
        ]]
        st.dataframe(display_df.style.format({
            "PRO_gap": "{:.4f}",
            "PRO融合模型_gap": "{:.4f}",
            "P(主胜)": "{:.4f}",
            "P(平局)": "{:.4f}",
            "P(客胜)": "{:.4f}",
            "融合信心": "{:.2f}",
            "冷门标准化得分": "{:.2f}",
            "推荐总分": "{:.2f}"
        }))

        # CSV 下载
        csv = result_df_sorted.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 下载预测结果 CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")

        st.session_state.result_df = result_df

