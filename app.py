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
.company-name { font-size:1.1em; font-weight:600;
  text-shadow:1px 1px 2px rgba(0,0,0,0.2);
}
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
    # 隐含概率转换
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    # 标签编码
    y = df["比赛结果"].map({"主胜":0, "平局":1, "客胜":2}).values
    # 手动指定权重: 主胜=1.0, 平局=1.05, 客胜=1.0
    w = np.array([1.0 if yi==0 else 1.05 if yi==1 else 1.0 for yi in y])
    model = HistGradientBoostingClassifier(
        learning_rate=0.01,
        max_depth=5,
        loss="log_loss",
        random_state=42
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
    uploaded = st.file_uploader("上传赔率文件 (Excel/CSV，每行15列)", type=["xlsx","csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state.input_df = df_up[feature_cols]
        st.success(f"✅ 已读取 {len(df_up)} 场比赛数据")
        st.dataframe(st.session_state.input_df)
else:
    st.subheader("🖊 手动录入 (逐公司一行)")
    with st.form("manual_form", clear_on_submit=True):
        inputs = {}
        for comp in company_order:
            c1, c2 = st.columns([1, 2])
            c1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inputs[comp] = c2.text_input("", placeholder="2.05 3.60 3.50", key=f"man_{comp}")
        if st.form_submit_button("提交录入"):
            vals = []
            for comp in company_order:
                parts = inputs[comp].split()
                vals.extend([float(x) for x in parts])
            st.session_state.input_df.loc[len(st.session_state.input_df)] = vals
            st.success("✅ 已添加 1 场比赛，输入框已清空")

# ──────────────── 历史相似比赛推荐 ────────────────
if not st.session_state.input_df.empty:
    st.subheader("🔍 历史相似比赛推荐")
    for idx, row in st.session_state.input_df.iterrows():
        # 使用隐含概率进行匹配
        inv = 1 / row.values.astype(float).reshape(5, 3)
        prob = inv / inv.sum(axis=1, keepdims=True)
        flat = prob.flatten().tolist()
        # 放宽阈值保证返回
        try:
            sims = st.session_state.matcher.match(flat, top_k=3, threshold=0.0)
        except TypeError:
            try:
                sims = st.session_state.matcher.match(flat, top_k=3)
            except Exception:
                sims = []
        except Exception:
            sims = []
        if sims:
            st.markdown(f"**第 {idx+1} 场** 最相似历史比赛：")
            st.table(pd.DataFrame(sims, columns=["历史序号", "相似度"]))
        else:
            st.markdown(f"**第 {idx+1} 场** 无可用历史匹配结果")

# ──────────────── 预测与融合 ────────────────
if not st.session_state.input_df.empty and st.button("🔄 运行综合预测"):
    df_in = st.session_state.input_df.copy().fillna(method="ffill")
    n = len(df_in)

    # 足球 HGB 模型概率
    odds = df_in.values.astype(float)
    Xs   = odds.copy()
    for j in range(0, 15, 3):
        inv = 1 / odds[:, j:j+3]
        Xs[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    p_soc = soccer_model.predict_proba(Xs)

    # PRO + PRO 融合
    df_pro = predict_model_pro(df_in)
    df_in["PRO_最终预测结果"]     = df_pro["最终预测结果"]
    df_in["PRO_gap"]              = df_pro["average_gap"]
    df_ens = predict_model_pro_ensemble(df_in)
    df_in["PRO融合模型预测结果"]   = df_ens["PRO融合模型预测结果"]
    df_in["PRO融合模型_gap"]       = df_ens["PRO融合模型_gap"]

    # META 融合模型概率
    try:
        df_meta = predict_model_meta(df_in)
        meta_cols = [f"P({o})" for o in outcomes]
        p_meta = df_meta[meta_cols].values if all(c in df_meta.columns for c in meta_cols) else np.zeros_like(p_soc)
    except Exception:
        p_meta = np.zeros_like(p_soc)

    # 最终融合概率：平均 + 归一化
    p_final = (p_soc + p_meta) / 2.0
    p_final = p_final / p_final.sum(axis=1, keepdims=True)
    preds   = [outcomes[i] for i in np.argmax(p_final, axis=1)]

    # 展示与下载
    res = pd.DataFrame(p_final * 100, columns=[f"{o}(%)" for o in outcomes])
    res.insert(0, "最终预测", preds)
    res.index = np.arange(1, n+1)
    res.index.name = "比赛编号"
    st.subheader("📊 综合模型预测结果")
    st.dataframe(res, use_container_width=True)
    csv = res.to_csv(index=True).encode("utf-8-sig")
    st.download_button("⬇ 下载结果", csv, "predictions.csv", "text/csv")
