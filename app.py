import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

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

# ───────── 加载多模型 ─────────
@st.cache_resource
def load_models():
    data_file = Path(__file__).parent / 'data' / 'new_matches.xlsx'
    df = pd.read_excel(data_file)
    feat_cols = [c for c in df.columns if c not in ['比赛','比赛结果']]
    X = df[feat_cols].values
    # 隐含概率转换
    X_imp = X.copy()
    for j in range(0, X_imp.shape[1], 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)
    # 原始标签
    y = df['比赛结果'].map({'主胜':0,'平局':1,'客胜':2}).values
    # 样本权重（平局增强）
    w = np.array([1.0 if yi==0 else 0.6 if yi==1 else 1.0 for yi in y])

    # 一阶段：平局 vs 非平局
    y_draw = (y == 1).astype(int)
    draw_clf = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss='log_loss', random_state=42
    )
    draw_clf.fit(X_imp, y_draw)
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X_imp, y_draw)

    # 二阶段：胜负模型（非平局子集）
    mask = (y != 1)
    X_wl, y_wl = X_imp[mask], (y[mask] == 0).astype(int)
    winlose_clf = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss='log_loss', random_state=42
    )
    winlose_clf.fit(X_wl, y_wl)

    # 备用三分类模型
    multi_clf = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss='log_loss', random_state=42
    )
    multi_clf.fit(X_imp, y, sample_weight=w)

    return feat_cols, draw_clf, tree_clf, winlose_clf, multi_clf

soccer_feats, draw_clf, tree_clf, winlose_clf, multi_clf = load_models()

# ───────── 页面设置 ─────────
st.set_page_config(page_title='二阶段+树平局 综合预测', layout='wide')
st.title('⚽ 二阶段 + 树 平局融合 预测器')

company_order = ['Bet365','立博','Interwetten','Pinnacle','William Hill']
outcomes      = ['主胜','平局','客胜']
feature_cols  = [f"{c}_{o}" for c in company_order for o in outcomes]

# SessionState 初始化
if 'input_df' not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=feature_cols)
if 'matcher' not in st.session_state:
    st.session_state.matcher = SimilarityMatcher(
        str(Path(__file__).parent / 'data' / 'prediction_results (43).xlsx')
    )

# ───────── 数据输入 ─────────
mode = st.radio('📥 数据输入方式', ['上传文件','手动录入'], horizontal=True)
if mode=='上传文件':
    up = st.file_uploader('上传赔率 (Excel/CSV，每行15列)', type=['xlsx','csv'])
    if up:
        df_up = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
        st.session_state.input_df = df_up[feature_cols]
        st.success(f'✅ 已读取 {len(df_up)} 场比赛')
        st.dataframe(st.session_state.input_df)
else:
    st.subheader('🖊 手动录入 (逐公司一行)')
    with st.form('frm_manual', clear_on_submit=True):
        inputs = {}
        for comp in company_order:
            c1,c2 = st.columns([1,2])
            c1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inputs[comp] = c2.text_input('', placeholder='2.05 3.60 3.50', key=f"man_{comp}")
        if st.form_submit_button('添加比赛'):
            vals=[]
            ok=True
            for comp in company_order:
                pts = inputs[comp].split()
                if len(pts)!=3:
                    st.error(f'⚠️ {comp} 需输入3个赔率')
                    ok=False
                    break
                vals.extend([float(x) for x in pts])
            if ok:
                new = pd.DataFrame([vals], columns=feature_cols)
                st.session_state.input_df = pd.concat([
                    st.session_state.input_df, new
                ], ignore_index=True)
                st.success('✅ 添加成功，输入框已清空')

# ───────── 历史匹配 ─────────
if not st.session_state.input_df.empty:
    st.subheader('🔍 历史相似比赛推荐')
    df_pro = predict_model_pro(st.session_state.input_df)
    prob_cols = [c for c in df_pro.columns if c.startswith('P(')]
    for pc in prob_cols:
        df_pro[pc].fillna(0, inplace=True)
    ens_in = pd.concat([
        st.session_state.input_df.reset_index(drop=True),
        df_pro[['average_gap']+prob_cols].reset_index(drop=True)
    ], axis=1)
    try:
        df_ens = predict_model_pro_ensemble(ens_in)
    except:
        df_ens = pd.DataFrame({
            'PRO融合模型预测结果':['平局']*len(df_pro),
            'PRO融合模型_gap':[0.0]*len(df_pro)
        })
    try:
        df_meta = predict_model_meta(st.session_state.input_df)
    except:
        df_meta = pd.DataFrame()
    for i in range(len(st.session_state.input_df)):
        q = {
            'PRO_gap':          df_pro.loc[i,'average_gap'],
            'PRO融合模型_gap':   df_ens.loc[i,'PRO融合模型_gap'],
            '融合信心':         df_meta.loc[i,'融合信心']   if '融合信心'  in df_meta else 0,
            '推荐总分':         df_meta.loc[i,'推荐总分']   if '推荐总分'  in df_meta else 0,
            'pair':            f"{df_pro.loc[i,'最终预测结果']}-{df_ens.loc[i,'PRO融合模型预测结果']}"
        }
        sims = pd.DataFrame()
        try:
            sims = st.session_state.matcher.query(q, k=3)
        except:
            pass
        st.markdown(f"**第 {i+1} 场** 历史相似比赛：")
        st.dataframe(sims.reset_index(drop=True))

# ───────── 二阶段+树 平局融合预测 ─────────
if not st.session_state.input_df.empty and st.button('🎯 运行预测'):
    df_in = st.session_state.input_df.copy().fillna(method='ffill')
    M = len(df_in)

    # 构造隐含概率特征
    odds     = df_in[soccer_feats].values.astype(float)
    X_imp_new = odds.copy()
    for j in range(0, X_imp_new.shape[1], 3):
        inv = 1 / X_imp_new[:, j:j+3]
        X_imp_new[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)

    # 1️⃣ 平局概率融合
    p_draw_hgb  = draw_clf.predict_proba(X_imp_new)[:,1]
    p_draw_tree = tree_clf.predict_proba(X_imp_new)[:,1]
    p_base_draw = 0.6 * p_draw_hgb + 0.4 * p_draw_tree
    # 非线性 boost (α=0.10, γ=0.50)
    alpha, gamma = 0.10, 0.50
    p_draw = p_base_draw + alpha * np.power(1 - p_base_draw, gamma)
    p_draw = np.clip(p_draw, 0, 1)

    # 2️⃣ 胜负概率
    p_wl   = winlose_clf.predict_proba(X_imp_new)
    p_base = np.zeros((M,3))
    p_base[:,1] = p_draw
    p_base[:,0] = p_wl[:,1] * (1 - p_draw)
    p_base[:,2] = p_wl[:,0] * (1 - p_draw)

    # 3️⃣ PRO+ensemble 概率
    df_pro2 = predict_model_pro(df_in)
    prob2   = [c for c in df_pro2.columns if c.startswith('P(')]
    for pc in prob2:
        df_pro2[pc].fillna(0, inplace=True)
    ens2_in = pd.concat([
        df_in.reset_index(drop=True),
        df_pro2[['average_gap']+prob2].reset_index(drop=True)
    ], axis=1)
    try:
        df_ens2 = predict_model_pro_ensemble(ens2_in)
        p_ens   = df_ens2[[f'P({o})' for o in outcomes]].values
    except:
        p_ens = np.zeros_like(p_base)

    # 4️⃣ META 概率
    try:
        df_meta2 = predict_model_meta(df_in)
        p_meta   = df_meta2[[f'P({o})' for o in outcomes]].values
    except:
        p_meta = np.zeros_like(p_base)

    # 5️⃣ 三路融合 + 初次归一化
    p_final = (p_base + p_ens + p_meta) / 3
    p_final /= p_final.sum(axis=1, keepdims=True)

    # 6️⃣ Upset-boost for 三种结果
    odds_mat    = odds.reshape(-1, len(company_order), len(outcomes))  # (M,5,3)
    inv_out     = 1 / odds_mat
    implied_out = inv_out.mean(axis=1)
    implied_out /= implied_out.sum(axis=1, keepdims=True)
    underdog_idx = np.argmin(implied_out, axis=1)

    for i, u in enumerate(underdog_idx):
        p_final[i, u] += alpha * np.power(1 - p_final[i, u], gamma)
    p_final /= p_final.sum(axis=1, keepdims=True)

    # 7️⃣ 输出结果
    preds  = [outcomes[i] for i in np.argmax(p_final, axis=1)]
    df_res = pd.DataFrame(p_final*100,
                          columns=[f'{o}(%)' for o in outcomes])
    df_res.insert(0, '最终预测', preds)
    df_res.index = np.arange(1, M+1)
    df_res.index.name = '比赛编号'

    st.subheader('📊 综合模型预测结果')
    st.dataframe(df_res, use_container_width=True)
    csv = df_res.to_csv(index=True).encode('utf-8-sig')
    st.download_button('⬇️ 下载结果', csv, 'predictions.csv', 'text/csv')
