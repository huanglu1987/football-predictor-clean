import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

# 导入 PRO 模型接口
from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble
# 导入 META 融合模型接口
from models.predict_model_meta import predict_model_meta
from similarity_matcher import SimilarityMatcher

# ──────────────── CSS Styling ────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0f8ff, #e6e6fa); }
.company-name { font-size:1.1em; font-weight:600; text-shadow:1px 1px 2px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { max-width:200px; }
.stButton>button { margin-top:4px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────── Load Soccer Predictor Model ────────────────
@st.cache_resource
def load_soccer_model():
    # 相对路径加载
    df = pd.read_excel(Path(__file__).parent / "data" / "new_matches.xlsx")
    cols = [c for c in df.columns if c not in ["比赛","比赛结果"]]
    X = df[cols].values
    # 隐含概率
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    y = df["比赛结果"].map({"主胜":0,"平局":1,"客胜":2}).values
    # 权重
    w = np.array([1.0 if yi==0 else 1.3 if yi==1 else 1.0 for yi in y])
    model = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    )
    model.fit(X_imp, y, sample_weight=w)
    return model, cols

soccer_model, soccer_feats = load_soccer_model()

# ──────────────── Streamlit Setup ────────────────
st.set_page_config(page_title="综合足球预测器", layout="wide")
st.title("⚽ 综合足球比赛预测器")

company_order = ["Bet365","立博","Interwetten","Pinnacle","William Hill"]
outcomes = ["主胜","平局","客胜"]
feature_cols = [f"{c}_{o}" for c in company_order for o in outcomes]

# Session state
if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=feature_cols)
if "matcher" not in st.session_state:
    st.session_state.matcher = SimilarityMatcher(str(Path(__file__).parent / "data" / "prediction_results (43).xlsx"))

# ──────────────── Data Input ────────────────
mode = st.radio("数据输入方式", ["📁 上传文件","🖊 手动录入"], horizontal=True)
if mode == "📁 上传文件":
    uploaded = st.file_uploader("上传赔率文件 (Excel/CSV，每行15列)", type=["xlsx","csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state.input_df = df_in[feature_cols]
        st.success(f"已读取 {len(df_in)} 场比赛数据")
        st.dataframe(st.session_state.input_df)
else:
    st.info("请按顺序粘贴各公司赔率: 主胜 平局 客胜，每完成5行点击提交")
    manual = {}
    for comp in company_order:
        manual[comp] = st.text_input(f"{comp}", placeholder="2.05 3.60 3.50", key=f"man_{comp}")
    if st.button("提交录入"):
        vals = []
        for comp in company_order:
            parts = manual[comp].split()
            vals.extend([float(x) for x in parts])
        st.session_state.input_df.loc[len(st.session_state.input_df)] = vals
        st.success("已添加1场比赛")

# ──────────────── Prediction & Integration ────────────────
if not st.session_state.input_df.empty and st.button("🔄 运行综合预测"):
    df_in = st.session_state.input_df.copy().fillna(method='ffill')
    n = len(df_in)
    # 1) 足球 HGB 模型 概率
    odds = df_in.values.astype(float)
    Xs = odds.copy()
    for j in range(0,15,3):
        inv = 1/odds[:, j:j+3]
        Xs[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    p_soc = soccer_model.predict_proba(Xs)
    # 2) PRO 模型 预测 和 概率    
    df_pro = predict_model_pro(df_in)
    df_in["PRO_最终预测结果"] = df_pro["最终预测结果"]
    df_in["PRO_gap"] = df_pro["average_gap"]
    # 3) PRO 融合 模型 预测
    df_ens = predict_model_pro_ensemble(df_in)
    df_in["PRO融合模型预测结果"] = df_ens["PRO融合模型预测结果"]
    df_in["PRO融合模型_gap"] = df_ens["PRO融合模型_gap"]
    # 4) META 融合 模型 输出 概率
    df_meta = predict_model_meta(df_in)
    p_meta = df_meta[[f"P({o})" for o in outcomes]].values
    # 5) 均值融合
    p_final = (p_soc + p_meta) / 2
    preds = [outcomes[i] for i in np.argmax(p_final, axis=1)]
    # 6) 展示结果
    res = pd.DataFrame(p_final*100, columns=[f"{o}(%)" for o in outcomes])
    res.insert(0, "最终预测", preds)
    res.index = np.arange(1, n+1)
    res.index.name = "比赛编号"
    st.subheader("综合模型预测结果")
    st.dataframe(res)
    csv = res.to_csv(index=True).encode('utf-8-sig')
    st.download_button("⬇ 下载结果", csv, "result.csv", "text/csv")
