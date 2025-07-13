import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

# 导入 PRO 和 META 模型接口
from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble
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

# ──────────────── 加载足球 HGB 模型 ────────────────
@st.cache_resource
def load_soccer_model():
    data_file = Path(__file__).parent / "data" / "new_matches.xlsx"
    df = pd.read_excel(data_file)
    feat_cols = [c for c in df.columns if c not in ["比赛", "比赛结果"]]
    X = df[feat_cols].values
    # 隐含概率
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    y = df["比赛结果"].map({"主胜": 0, "平局": 1, "客胜": 2}).values
    # 手动权重 (示例用 1.3，若需调整可修改此处)
    w = np.array([1.0 if yi==0 else 1.3 if yi==1 else 1.0 for yi in y])
    model = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    )
    model.fit(X_imp, y, sample_weight=w)
    return model, feat_cols

soccer_model, soccer_feats = load_soccer_model()

# ──────────────── Streamlit 页面设置 ────────────────
st.set_page_config(page_title="综合足球预测器", layout="wide")
st.title("⚽ 综合足球比赛预测器")

company_order = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
outcomes      = ["主胜", "平局", "客胜"]
feature_cols  = [f"{c}_{o}" for c in company_order for o in outcomes]

# Session state 初始化
if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=feature_cols)
if "matcher" not in st.session_state:
    hist_file = Path(__file__).parent / "data" / "prediction_results (43).xlsx"
    st.session_state.matcher = SimilarityMatcher(str(hist_file))

# ──────────────── 数据输入 ────────────────
mode = st.radio("数据输入方式", ["📁 上传文件", "🖊 手动录入"], horizontal=True)
if mode == "📁 上传文件":
    uploaded = st.file_uploader("上传赔率文件 (Excel/CSV，每行15列)", type=["xlsx", "csv"])
    if uploaded:
        df_up = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                 else pd.read_excel(uploaded))
        st.session_state.input_df = df_up[feature_cols]
        st.success(f"✅ 已读取 {len(df_up)} 场比赛数据")
        st.dataframe(st.session_state.input_df)
else:
    st.info("请按顺序粘贴各公司赔率：主胜 平局 客胜，输入完 5 行后点击【提交录入】")
    manual = {}
    for comp in company_order:
        manual[comp] = st.text_input(comp, placeholder="2.05 3.60 3.50", key=f"man_{comp}")
    if st.button("提交录入"):
        vals = []
        for comp in company_order:
            parts = manual[comp].split()
            vals.extend([float(x) for x in parts])
        st.session_state.input_df.loc[len(st.session_state.input_df)] = vals
        st.success("✅ 已添加 1 场比赛")

# ──────────────── 预测与融合 ────────────────
if not st.session_state.input_df.empty and st.button("🔄 运行综合预测"):
    df_in = st.session_state.input_df.copy().fillna(method="ffill")
    n = len(df_in)

    # 1) 足球 HGB 模型概率
    odds = df_in.values.astype(float)
    Xs = odds.copy()
    for j in range(0, 15, 3):
        inv = 1 / odds[:, j:j+3]
        Xs[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    p_soc = soccer_model.predict_proba(Xs)

    # 2) PRO & PRO 融合 模型（示例：可以同时提取它们的概率或得分）
    df_pro = predict_model_pro(df_in)
    df_ens = predict_model_pro_ensemble(df_in)
    # 假设它们输出的概率列为 P(主胜)、P(平局)、P(客胜)
    p_pro = df_pro[[f"P({o})" for o in outcomes]].values if set([f"P({o})" for o in outcomes]).issubset(df_pro.columns) else np.zeros((n,3))
    p_ens = df_ens[[f"P({o})" for o in outcomes]].values if set([f"P({o})" for o in outcomes]).issubset(df_ens.columns) else np.zeros((n,3))

    # 3) META 融合模型概率，并做 KeyError 自动补齐
    try:
        df_meta = predict_model_meta(df_in)
    except KeyError as e:
        # 自动从异常信息中提取缺失列名并补 0
        import re
        msg = str(e)
        missing = re.findall(r"\\['([^']+)'\\]", msg)
        for col in missing:
            df_in[col] = 0
        df_meta = predict_model_meta(df_in)
    p_meta = df_meta[[f"P({o})" for o in outcomes]].values

    # 4) 最终取平均融合（可根据业务调权重）
    p_final = (p_soc + p_pro + p_ens + p_meta) / 4.0
    preds = [outcomes[i] for i in np.argmax(p_final, axis=1)]

    # 5) 展示结果
    res = pd.DataFrame(p_final * 100, columns=[f"{o}(%)" for o in outcomes])
    res.insert(0, "最终预测", preds)
    res.index = np.arange(1, n+1)
    res.index.name = "比赛编号"

    st.subheader("📊 综合模型预测结果")
    st.dataframe(res, use_container_width=True)
    csv = res.to_csv(index=True).encode("utf-8-sig")
    st.download_button("⬇️ 下载结果 CSV", csv, "predictions.csv", "text/csv")
