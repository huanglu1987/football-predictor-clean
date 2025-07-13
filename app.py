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

# ───────── CSS Styling ─────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0f8ff, #e6e6fa); }
.company-name { font-size:1.1em; font-weight:600; text-shadow:1px 1px 2px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { max-width:200px; }
.stButton>button { margin-top:4px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ───────── 加载 足球 HGB 模型 ─────────
@st.cache_resource
def load_soccer_model():
    df = pd.read_excel(Path(__file__).parent / 'data' / 'new_matches.xlsx')
    feat_cols = [c for c in df.columns if c not in ['比赛','比赛结果']]
    X = df[feat_cols].values
    # 隐含概率转换
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    y = df['比赛结果'].map({'主胜':0,'平局':1,'客胜':2}).values
    # 权重：平局=0.70
    w = np.array([1.0 if yi==0 else 0.70 if yi==1 else 1.0 for yi in y])
    model = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss='log_loss', random_state=42
    )
    model.fit(X_imp, y, sample_weight=w)
    return model, feat_cols

soccer_model, soccer_feats = load_soccer_model()

# ───────── 页面设置 ─────────
st.set_page_config(page_title='综合足球预测器', layout='wide')
st.title('⚽ 综合足球比赛预测器')

company_order = ['Bet365','立博','Interwetten','Pinnacle','William Hill']
outcomes      = ['主胜','平局','客胜']
feature_cols  = [f"{c}_{o}" for c in company_order for o in outcomes]
num_features  = len(feature_cols)

# Session state 初始化
if 'input_df' not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=feature_cols)
if 'matcher' not in st.session_state:
    st.session_state.matcher = SimilarityMatcher(
        str(Path(__file__).parent / 'data' / 'prediction_results (43).xlsx')
    )

# ───────── 数据输入 ─────────
mode = st.radio('数据输入方式', ['📁 上传文件','🖊 手动录入'], horizontal=True)
if mode=='📁 上传文件':
    up = st.file_uploader('上传赔率文件 (Excel/CSV，每行15列)', type=['xlsx','csv'])
    if up:
        df_up = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
        st.session_state.input_df = df_up[feature_cols]
        st.success(f'✅ 已读取 {len(df_up)} 场比赛')
        st.dataframe(st.session_state.input_df)
else:
    st.subheader('🖊 手动录入 (逐公司一行)')
    with st.form('manual_form', clear_on_submit=True):
        inputs = {}
        for comp in company_order:
            c1,c2 = st.columns([1,2])
            c1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inputs[comp] = c2.text_input('', placeholder='2.05 3.60 3.50', key=f'man_{comp}')
        if st.form_submit_button('提交录入'):
            vals=[]
            for comp in company_order:
                parts = inputs[comp].split()
                if len(parts)!=3:
                    st.error(f'⚠️ {comp} 需3个赔率，当前{len(parts)}')
                    vals=None
                    break
                vals.extend([float(x) for x in parts])
            if vals and len(vals)==num_features:
                new_row = pd.DataFrame([vals], columns=feature_cols)
                st.session_state.input_df = pd.concat(
                    [st.session_state.input_df, new_row], ignore_index=True
                )
                st.success('✅ 已添加1场比赛，输入框已清空')

# ───────── 历史匹配 ─────────
if not st.session_state.input_df.empty:
    st.subheader('🔍 历史相似比赛推荐')
    df_pro = predict_model_pro(st.session_state.input_df)
    df_ens = None
    # 构造 ensemble 输入，动态提取 PRO 概率列
    prob_cols = [c for c in df_pro.columns if c.startswith('P(')]
    for pc in prob_cols:
        df_pro[pc] = df_pro[pc].fillna(0)
    ens_input = pd.concat([
        st.session_state.input_df.reset_index(drop=True),
        df_pro[['average_gap']+prob_cols].reset_index(drop=True)
    ], axis=1)
    try:
        df_ens = predict_model_pro_ensemble(ens_input)
    except Exception:
        df_ens = pd.DataFrame({
            'PRO融合模型预测结果': ['平局']*len(df_pro),
            'PRO融合模型_gap': [0.0]*len(df_pro)
        })
    try:
        df_meta = predict_model_meta(st.session_state.input_df)
    except:
        df_meta = pd.DataFrame()
    for idx in range(len(st.session_state.input_df)):
        query = {
            'PRO_gap':         df_pro.loc[idx,'average_gap'],
            'PRO融合模型_gap': df_ens.loc[idx,'PRO融合模型_gap'],
            '融合信心':        df_meta.loc[idx,'融合信心']   if '融合信心' in df_meta else 0,
            '推荐总分':        df_meta.loc[idx,'推荐总分']   if '推荐总分' in df_meta else 0,
            'pair':            f"{df_pro.loc[idx,'最终预测结果']}-{df_ens.loc[idx,'PRO融合模型预测结果']}"
        }
        try:
            sims = st.session_state.matcher.query(query, k=3)
        except:
            sims = pd.DataFrame()
        st.markdown(f"**第{idx+1}场** 最相似历史比赛：")
        st.dataframe(sims)

# ───────── 预测与融合 ─────────
if not st.session_state.input_df.empty and st.button('🔄 运行综合预测'):
    df_in = st.session_state.input_df.copy().fillna(method='ffill')
    n = len(df_in)
    X_odds = df_in[soccer_feats].values.astype(float)
    Xs = X_odds.copy()
    for j in range(0, Xs.shape[1], 3):
        inv = 1 / Xs[:,j:j+3]
        Xs[:,j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    p_soc = soccer_model.predict_proba(Xs)
    # PRO ensemble
    df_pro2 = predict_model_pro(df_in)
    prob_cols2 = [c for c in df_pro2.columns if c.startswith('P(')]
    for pc in prob_cols2:
        df_pro2[pc] = df_pro2[pc].fillna(0)
    ens2_input = pd.concat([
        df_in.reset_index(drop=True),
        df_pro2[['average_gap']+prob_cols2].reset_index(drop=True)
    ], axis=1)
    try:
        df_ens2 = predict_model_pro_ensemble(ens2_input)
        p_ens = df_ens2[[f'P({o})' for o in outcomes]].values
    except Exception:
        p_ens = np.zeros_like(p_soc)
    # META probabilities
    try:
        df_meta2 = predict_model_meta(df_in)
        p_meta = df_meta2[[f'P({o})' for o in outcomes]].values
    except:
        p_meta = np.zeros_like(p_soc)
    # 融合+归一化
    p_final = (p_soc + p_ens + p_meta) / 3
    p_final /= p_final.sum(axis=1, keepdims=True)
    preds = [outcomes[i] for i in np.argmax(p_final, axis=1)]
    res = pd.DataFrame(p_final*100, columns=[f'{o}(%)' for o in outcomes])
    res.insert(0,'最终预测',preds)
    res.index = np.arange(1,n+1)
    res.index.name='比赛编号'
    st.subheader('📊 综合模型预测结果')
    st.dataframe(res, use_container_width=True)
    csv = res.to_csv(index=True).encode('utf-8-sig')
    st.download_button('⬇ 下载结果', csv, 'predictions.csv', 'text/csv')
