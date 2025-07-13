import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

# ──────────────── CSS Styling ────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0f8ff, #e6e6fa); }
.company-name { font-size:1.1em; font-weight:600; text-shadow:1px 1px 2px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { max-width:200px; }
.stButton>button { margin-top:4px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────── Load Soccer Predictor ────────────────
@st.cache_resource
def load_soccer_model():
    df = pd.read_excel("data/new_matches.xlsx")
    cols = [c for c in df.columns if c not in ["比赛","比赛结果"]]
    X = df[cols].values
    # implicit prob
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    y = df["比赛结果"].map({"主胜":0,"平局":1,"客胜":2}).values
    # weights
    w = np.array([1.0 if yi==0 else 1.3 if yi==1 else 1.0 for yi in y])
    model = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    )
    model.fit(X_imp, y, sample_weight=w)
    return model, cols

soccer_model, soccer_feats = load_soccer_model()

# ──────────────── META Model Imports ────────────────
from models.predict_model_meta import predict_model_meta

# ──────────────── App Config ────────────────
st.set_page_config(page_title="综合足球预测器", layout="wide")
st.title("⚽ 综合足球比赛预测器")

# features and outcomes
company_order = ["Bet365","立博","Interwetten","Pinnacle","William Hill"]
outcomes = ["主胜","平局","客胜"]
feature_cols = [f"{c}_{o}" for c in company_order for o in outcomes]

# session state for input data
if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=feature_cols)

# ──────────────── Data Input ────────────────
mode = st.radio("数据输入方式", ["📁上传文件","🖊手动录入"], horizontal=True)
if mode=="📁上传文件":
    uploaded = st.file_uploader("上传15列赔率文件", type=["xlsx","csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state.input_df = df_in[feature_cols]
        st.success(f"已上传 {len(df_in)} 场比赛")
        st.dataframe(st.session_state.input_df)
else:
    st.info("按照顺序粘贴每家公司赔率: 主胜 平局 客胜，每完成5行点击提交")
    manual = []
    for comp in company_order:
        odds = st.text_input(comp, placeholder="2.05 3.60 3.50", key=f"man_{comp}")
        manual.append(odds)
    if st.button("提交手动录入"):
        vals = []
        for o in manual:
            parts = o.split()
            vals.extend([float(x) for x in parts])
        st.session_state.input_df.loc[len(st.session_state.input_df)] = vals
        st.success("已添加1场比赛")

# ──────────────── Prediction ────────────────
if not st.session_state.input_df.empty:
    if st.button("🔄 运行综合预测"):
        df_in = st.session_state.input_df.fillna(method='ffill')
        n = len(df_in)
        # 1) Soccer predictor probabilities
        odds = df_in.values.astype(float)
        Xs = odds.copy()
        for j in range(0,15,3):
            inv = 1 / odds[:, j:j+3]
            Xs[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
        p1 = soccer_model.predict_proba(Xs)
        # 2) META model probabilities
        df_meta = predict_model_meta(df_in)
        p2 = df_meta[[f"P({o})" for o in outcomes]].values
        # 3) average
        pavg = (p1 + p2) / 2
        preds = [outcomes[i] for i in np.argmax(pavg, axis=1)]
        # results df
        res = pd.DataFrame(pavg*100, columns=[f"{o}(%)" for o in outcomes])
        res.insert(0, "最终预测", preds)
        res.index = np.arange(1,n+1)
        res.index.name = "比赛编号"
        st.subheader("综合模型预测结果")
        st.dataframe(res)
        csv = res.to_csv().encode('utf-8-sig')
        st.download_button("⬇下载结果", csv, "result.csv", "text/csv")
