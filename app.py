# app.py  –  云端 & 本地统一版 + 系统推荐逻辑（阈值 0.03）
# 功能：
# - CV / Range / Pdiff 特征
# - 历史相似比赛推荐（沿用原逻辑）
# - 基于5场相似比赛计算当前比赛的6个模式（PRO / PRO融合）
# - 使用6个模式单独匹配全库（模式匹配参考）
# - 系统推荐结果（主选 + 备选，阈值=0.03）
# - 导出新比赛的模式

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# ───────── 外部模型接口 ─────────
from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble
from models.predict_model_meta import predict_model_meta
from similarity_matcher import SimilarityMatcher

# ───────── UI 样式 ─────────
st.set_page_config(page_title="CV/Range/Pdiff Boost 预测", layout="wide")

st.markdown("""
<style>
.stApp { background:linear-gradient(135deg,#f0f8ff,#e6e6fa); }
.company-name{font-size:1.1em;font-weight:600;text-shadow:1px 1px 2px rgba(0,0,0,0.2);}
.stTextInput>div>div>input{max-width:200px;}
.stButton>button{margin-top:4px;margin-bottom:8px;}

/* gap 差值模式表格：两列并排、行内对齐 */
.gap-table{
  width: auto;
  margin: 0.5rem auto 0.8rem auto;
  border-collapse: separate;
  border-spacing: 15px 4px;
}
.gap-table th{
  text-align: left;
  font-weight: 600;
  padding-bottom: 4px;
}
.gap-table td{
  padding: 2px 0;
  font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ───────── 基本常量（相对路径，云端 & 本地统一） ─────────
BASE_DIR  = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "new_matches.xlsx"
HIST_PATH = BASE_DIR / "data" / "prediction_results (43).xlsx"

companies = ["Bet365","立博","Interwetten","Pinnacle","William Hill"]
outcomes  = ["主胜","平局","客胜"]
ODDS_COLS = [f"{c}_{o}" for c in companies for o in outcomes]  # 15 列
OUTCOME_COL = "比赛结果"  # 历史库中真实结果列名

# ───────── 1. 衍生特征函数 ─────────
def add_extra_features(df_odds: pd.DataFrame) -> pd.DataFrame:
    extra = pd.DataFrame(index=df_odds.index)

    mat = df_odds.values.reshape(-1, len(companies), len(outcomes))  # (N,5,3)

    cv_vals    = np.std(mat, axis=1) / np.mean(mat, axis=1)
    range_vals = np.max(mat, axis=1) - np.min(mat, axis=1)
    extra[["cv_home","cv_draw","cv_away"]]           = cv_vals
    extra[["range_home","range_draw","range_away"]] = range_vals

    inv = 1 / mat
    imp = inv / inv.sum(axis=2, keepdims=True)
    p_diff = np.abs(imp[:,:,0].mean(axis=1) - imp[:,:,2].mean(axis=1))
    extra["p_diff"] = p_diff

    return pd.concat(
        [df_odds.reset_index(drop=True), extra.reset_index(drop=True)],
        axis=1
    )

# ───────── 1.1 gap 差值模式工具函数 ─────────
def _truncate_two_decimals(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x >= 0:
        return math.floor(x * 100 + 1e-8) / 100.0
    else:
        return math.ceil(x * 100 - 1e-8) / 100.0


def format_gap_pattern(values) -> str:
    if len(values) != 3:
        return ""
    vs = sorted(_truncate_two_decimals(v) for v in values)
    d1 = int(math.floor(abs(vs[1] - vs[0]) * 100 + 1e-8))
    d2 = int(math.floor(abs(vs[2] - vs[1]) * 100 + 1e-8))

    if (d1 == 0 and d2 != 0) or (d2 == 0 and d1 != 0):
        first, second = 0, d2 if d1 == 0 else d1
    else:
        first, second = d1, d2

    outer = abs(first - second)
    return f"({outer}){first}-{second}"


def compute_gap_patterns(sims: pd.DataFrame, col: str) -> dict:
    if col not in sims.columns:
        return {}
    vals = sims[col].tolist()
    windows = [("0-1-2", 0), ("1-2-3", 1), ("2-3-4", 2)]
    patterns = {}
    for label, start in windows:
        if len(vals) >= start + 3:
            triple = vals[start:start+3]
            patterns[label] = format_gap_pattern(triple)
    return patterns

# ───────── 1.2 系统推荐逻辑（单场） ─────────
def system_total_scores_for_match(
    i: int,
    df_pro: pd.DataFrame,
    df_ens: pd.DataFrame,
    sims_basic: pd.DataFrame,
    sims_pattern: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
) -> dict:
    """
    对第 i 场比赛，计算“系统推荐逻辑”下每个结果（主胜/平局/客胜）的综合得分。
    逻辑与你评估脚本一致：模型 + 历史相似 + 模式强参考。
    """
    # 1) 模型分数：0.4 * PRO + 0.6 * PRO融合
    model_score = {o: 0.0 for o in outcomes}
    for o in outcomes:
        p_pro = df_pro.loc[i, f"P({o})"] if f"P({o})" in df_pro.columns else 0.0
        p_ens = 0.0
        if df_ens is not None and f"P({o})" in df_ens.columns:
            p_ens = df_ens.loc[i, f"P({o})"]
        model_score[o] = 0.4 * float(p_pro) + 0.6 * float(p_ens)

    # 2) 历史相似比赛投票（sims_basic，一般5场）
    hist_basic_votes = {o: 0.0 for o in outcomes}
    if not sims_basic.empty and outcome_col in sims_basic.columns:
        for idx, row in sims_basic.iterrows():
            res = str(row[outcome_col])
            if res in hist_basic_votes:
                hist_basic_votes[res] += 1.0 / (1.0 + idx)  # 1, 1/2, 1/3...

    max_basic = max(hist_basic_votes.values()) if any(hist_basic_votes.values()) else 1.0
    hist_basic_score = {o: (hist_basic_votes[o] / max_basic) for o in outcomes}

    # 3) 模式匹配参考投票（只用强参考行）
    hist_pattern_votes = {o: 0.0 for o in outcomes}
    if not sims_pattern.empty and outcome_col in sims_pattern.columns:
        for idx, row in sims_pattern.iterrows():
            if not bool(row.get("强参考", False)):
                continue
            res = str(row[outcome_col])
            if res in hist_pattern_votes:
                hist_pattern_votes[res] += 1.0 / (1.0 + idx)

    max_pattern = max(hist_pattern_votes.values()) if any(hist_pattern_votes.values()) else 1.0
    hist_pattern_score = {o: (hist_pattern_votes[o] / max_pattern) for o in outcomes}

    # 4) 综合得分（与你评估脚本一致的权重）
    w_model   = 0.5
    w_basic   = 0.3
    w_pattern = 0.2

    total_score = {}
    for o in outcomes:
        total_score[o] = (
            w_model   * model_score[o] +
            w_basic   * hist_basic_score[o] +
            w_pattern * hist_pattern_score[o]
        )

    return total_score


def system_recommendation_for_match(
    i: int,
    df_pro: pd.DataFrame,
    df_ens: pd.DataFrame,
    sims_basic: pd.DataFrame,
    sims_pattern: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    threshold: float = 0.03,  # 单/双选阈值
) -> dict:
    """
    基于 total_score + 阈值，给出：
        - 主选
        - 备选（若差值 < 阈值）
        - total_score 字典
    """
    total_score = system_total_scores_for_match(
        i=i,
        df_pro=df_pro,
        df_ens=df_ens,
        sims_basic=sims_basic,
        sims_pattern=sims_pattern,
        outcome_col=outcome_col,
    )

    ordered = sorted(outcomes, key=lambda o: total_score[o], reverse=True)
    best   = ordered[0]
    second = ordered[1]

    diff = total_score[best] - total_score[second]
    if diff < threshold:
        backup = second
    else:
        backup = None

    return {
        "主选": best,
        "备选": backup,
        "total_score": total_score,
    }

# ───────── 2. 训练并缓存模型 ─────────
@st.cache_resource
def load_models():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")
    df_raw = pd.read_excel(DATA_FILE)
    X_base = df_raw[ODDS_COLS]
    X_feat = add_extra_features(X_base)          # 15+9 = 24 列
    feat_cols = X_feat.columns.tolist()

    X_imp = X_feat.copy().values.astype(float)
    for j in range(0, 15, 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)

    y = df_raw["比赛结果"].map({"主胜":0,"平局":1,"客胜":2}).values

    y_draw = (y == 1).astype(int)
    draw_hgb = HistGradientBoostingClassifier(
        learning_rate=0.01,max_depth=5,loss="log_loss",random_state=42
    ).fit(X_imp, y_draw)
    draw_tree = DecisionTreeClassifier(max_depth=3,random_state=42).fit(X_imp, y_draw)

    mask = (y != 1)
    X_wl, y_wl = X_imp[mask], (y[mask] == 0).astype(int)
    winlose_hgb = HistGradientBoostingClassifier(
        learning_rate=0.01,max_depth=5,loss="log_loss",random_state=42
    ).fit(X_wl, y_wl)

    class_w = {0:1.0, 1:0.8, 2:1.0}
    samp_w  = np.array([class_w[yi] for yi in y])
    multi_clf = HistGradientBoostingClassifier(
        learning_rate=0.01,max_depth=5,loss="log_loss",random_state=42
    ).fit(X_imp, y, sample_weight=samp_w)

    return feat_cols, draw_hgb, draw_tree, winlose_hgb, multi_clf

feat_cols, draw_hgb, draw_tree, winlose_hgb, multi_clf = load_models()

# ───────── 3. Streamlit 页面 ─────────
st.title("⚽足球预测系统")

if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=ODDS_COLS)
if "matcher" not in st.session_state:
    if not HIST_PATH.exists():
        st.error(f"找不到历史结果文件：{HIST_PATH}")
    else:
        st.session_state.matcher = SimilarityMatcher(str(HIST_PATH))
if "pattern_records" not in st.session_state:
    st.session_state.pattern_records = []

# ---------- 数据输入 ----------
mode = st.radio("📥 数据输入方式", ["上传文件","手动录入"], horizontal=True)

if mode == "上传文件":
    up = st.file_uploader("上传赔率文件 (Excel/CSV，每行15列)", type=["xlsx","csv"])
    if up is not None:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.session_state.input_df = df_up[ODDS_COLS]
        st.session_state.pattern_records = []
        st.success(f"✅ 已读取 {len(df_up)} 场比赛")
        st.dataframe(st.session_state.input_df)

else:
    st.subheader("🖊 手动录入 (逐公司一行)")
    with st.form("manual", clear_on_submit=True):
        inps = {}
        for comp in companies:
            c1,c2 = st.columns([1,2])
            c1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inps[comp] = c2.text_input("", placeholder="2.05 3.60 3.50", key=f"man_{comp}")
        if st.form_submit_button("添加比赛"):
            row, ok = [], True
            for comp in companies:
                parts = inps[comp].split()
                if len(parts)!=3:
                    st.error(f"{comp} 需输入 3 个赔率"); ok=False; break
                row += [float(x) for x in parts]
            if ok:
                st.session_state.input_df = pd.concat(
                    [st.session_state.input_df,
                     pd.DataFrame([row], columns=ODDS_COLS)],
                    ignore_index=True
                )
                st.session_state.pattern_records = []
                st.success("✅ 已添加1场比赛")

# ---------- 历史匹配 + 模式匹配参考 + 系统推荐 ----------
if not st.session_state.input_df.empty and "matcher" in st.session_state:
    st.subheader("🔍 历史相似比赛推荐 & 模式匹配参考 & 系统推荐")

    df_pro    = predict_model_pro(st.session_state.input_df)
    prob_cols = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob_cols: df_pro[pc].fillna(0, inplace=True)

    ens_in = pd.concat([
        st.session_state.input_df.reset_index(drop=True),
        df_pro[["average_gap"] + prob_cols].reset_index(drop=True)
    ], axis=1)
    try:
        df_ens = predict_model_pro_ensemble(ens_in)
    except:
        df_ens = pd.DataFrame({
            "PRO融合模型预测结果": ["平局"]*len(df_pro),
            "PRO融合模型_gap": [0.0]*len(df_pro)
        })
    try:
        df_meta = predict_model_meta(st.session_state.input_df)
    except:
        df_meta = pd.DataFrame()

    st.session_state.pattern_records = []

    for i in range(len(st.session_state.input_df)):
        st.markdown(f"### ▶ 第 {i+1} 场")

        # 1) 历史相似比赛推荐（原逻辑）
        q_basic = {
            "PRO_gap": df_pro.loc[i,"average_gap"],
            "PRO融合模型_gap": df_ens.loc[i,"PRO融合模型_gap"],
            "融合信心": df_meta.loc[i,"融合信心"] if "融合信心" in df_meta else 0,
            "推荐总分": df_meta.loc[i,"推荐总分"] if "推荐总分" in df_meta else 0,
            "pair": f"{df_pro.loc[i,'最终预测结果']}-{df_ens.loc[i,'PRO融合模型预测结果']}"
        }
        try:
            sims_basic = st.session_state.matcher.query(q_basic, k=5)
        except Exception as e:
            st.warning(f"历史匹配调用出错：{e}")
            sims_basic = pd.DataFrame()

        st.markdown("**1️⃣ 历史相似比赛（原逻辑）**")
        sims_basic = sims_basic.reset_index(drop=True)
        st.dataframe(sims_basic, use_container_width=True)

        # 2) 用这5场算当前比赛的6个模式
        pro_patterns = compute_gap_patterns(sims_basic, "PRO_gap")
        ens_patterns = compute_gap_patterns(sims_basic, "PRO融合模型_gap")

        pro0 = pro_patterns.get("0-1-2", "")
        pro1 = pro_patterns.get("1-2-3", "")
        pro2 = pro_patterns.get("2-3-4", "")

        ens0 = ens_patterns.get("0-1-2", "")
        ens1 = ens_patterns.get("1-2-3", "")
        ens2 = ens_patterns.get("2-3-4", "")

        if pro0 or pro1 or pro2 or ens0 or ens1 or ens2:
            st.markdown("**2️⃣ 当前比赛的 PRO / PRO融合 差值模式**")
            html = f"""
<table class="gap-table">
  <tr>
    <th>PRO_gap 差值模式</th>
    <th>PRO融合模型_gap 差值模式</th>
  </tr>
  <tr>
    <td>{pro0}</td>
    <td>{ens0}</td>
  </tr>
  <tr>
    <td>{pro1}</td>
    <td>{ens1}</td>
  </tr>
  <tr>
    <td>{pro2}</td>
    <td>{ens2}</td>
  </tr>
</table>
"""
            st.markdown(html, unsafe_allow_html=True)

        st.session_state.pattern_records.append({
            "比赛编号": i+1,
            "PRO_pattern_0_1_2": pro0,
            "PRO_pattern_1_2_3": pro1,
            "PRO_pattern_2_3_4": pro2,
            "ENS_pattern_0_1_2": ens0,
            "ENS_pattern_1_2_3": ens1,
            "ENS_pattern_2_3_4": ens2,
        })

        # 3) 模式匹配参考：只基于6个模式字段匹配全库
        q_pattern = {
            "PRO_pattern_0_1_2": pro0,
            "PRO_pattern_1_2_3": pro1,
            "PRO_pattern_2_3_4": pro2,
            "ENS_pattern_0_1_2": ens0,
            "ENS_pattern_1_2_3": ens1,
            "ENS_pattern_2_3_4": ens2,
        }
        try:
            sims_pattern = st.session_state.matcher.query(q_pattern, k=15)
        except Exception as e:
            st.warning(f"模式匹配调用出错：{e}")
            sims_pattern = pd.DataFrame()

        st.markdown("**3️⃣ 模式匹配参考（仅按6个模式匹配全库）**")
        if not sims_pattern.empty:
            sims_pattern = sims_pattern.copy()
            if "强参考" in sims_pattern.columns:
                sims_pattern["强参考标记"] = sims_pattern["强参考"].map(
                    lambda x: "⭐ 强参考" if bool(x) else ""
                )
            if "_distance" in sims_pattern.columns:
                sims_pattern = sims_pattern.drop(columns=["_distance"])
            st.dataframe(sims_pattern, use_container_width=True)
        else:
            st.info("暂无模式匹配结果（可能是模式或历史库配置有问题）。")

        # 4) 系统推荐结果（主选 + 备选，阈值=0.03）
        try:
            rec = system_recommendation_for_match(
                i=i,
                df_pro=df_pro,
                df_ens=df_ens,
                sims_basic=sims_basic,
                sims_pattern=sims_pattern,
                outcome_col=OUTCOME_COL,
                threshold=0.03,  # 你评估后选定的阈值
            )
            main_pick = rec["主选"]
            backup_pick = rec["备选"]
            scores = rec["total_score"]

            st.markdown("**4️⃣ 系统推荐结果（综合模型 + 历史 + 模式）**")
            if backup_pick:
                st.write(f"系统推荐主选：**{main_pick}** ，备选：**{backup_pick}**")
            else:
                st.write(f"系统推荐主选：**{main_pick}**（暂不给出备选）")

            score_str = " | ".join([f"{o}: {scores[o]:.3f}" for o in outcomes])
            st.caption(f"总分细节：{score_str}")
        except Exception as e:
            st.info(f"系统推荐计算出错（可暂时忽略）：{e}")

    # 5) 差值模式导出
    if st.session_state.pattern_records:
        df_patterns = pd.DataFrame(st.session_state.pattern_records).sort_values("比赛编号")
        st.subheader("📤 差值模式导出（每场新比赛的 PRO / PRO融合 差值模式）")
        st.dataframe(df_patterns, use_container_width=True)
        st.download_button(
            "⬇️ 下载差值模式（CSV）",
            df_patterns.to_csv(index=False).encode("utf-8-sig"),
            "gap_patterns_export_new_matches.csv",
            "text/csv",
        )

# ---------- 预测 ----------
if not st.session_state.input_df.empty and st.button("🎯 运行预测"):
    df_odds  = st.session_state.input_df.copy()
    X_feat   = add_extra_features(df_odds)
    X_imp    = X_feat.values.astype(float)
    for j in range(0, 15, 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)

    p_draw = 0.6*draw_hgb.predict_proba(X_imp)[:,1] + 0.4*draw_tree.predict_proba(X_imp)[:,1]
    p_draw = np.clip(p_draw + 0.10*np.power(1-p_draw,0.50), 0, 1)

    p_wl   = winlose_hgb.predict_proba(X_imp)
    p_base = np.zeros((len(X_imp),3))
    p_base[:,1] = p_draw
    p_base[:,0] = p_wl[:,1]*(1-p_draw)
    p_base[:,2] = p_wl[:,0]*(1-p_draw)

    df_pro  = predict_model_pro(df_odds)
    prob2   = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob2: df_pro[pc].fillna(0, inplace=True)
    ens_in  = pd.concat([df_odds.reset_index(drop=True),
                         df_pro[["average_gap"]+prob2].reset_index(drop=True)], axis=1)
    try:
        df_ens = predict_model_pro_ensemble(ens_in)
        p_ens  = df_ens[[f"P({o})" for o in outcomes]].values
    except:
        p_ens = np.zeros_like(p_base)

    try:
        df_meta = predict_model_meta(df_odds)
        p_meta  = df_meta[[f"P({o})" for o in outcomes]].values
    except:
        p_meta = np.zeros_like(p_base)

    p_multi = multi_clf.predict_proba(X_imp)

    w_base, w_ens, w_meta, w_multi = 0.10, 0.70, 0.10, 0.00
    wsum   = w_base + w_ens + w_meta + w_multi

    p_final = (w_base  * p_base  +
               w_ens   * p_ens   +
               w_meta  * p_meta  +
               w_multi * p_multi) / wsum

    p_final /= p_final.sum(axis=1, keepdims=True)

    preds = [outcomes[k] for k in p_final.argmax(axis=1)]
    df_res = pd.DataFrame(p_final*100, columns=[f"{o}(%)" for o in outcomes])
    df_res.insert(0,"最终预测", preds)
    df_res.index = np.arange(1,len(df_res)+1); df_res.index.name="比赛编号"

    st.subheader("📊 综合模型预测结果")
    st.dataframe(df_res, use_container_width=True)
    st.download_button(
        "⬇️ 下载结果",
        df_res.to_csv(index=True).encode("utf-8-sig"),
        "predictions.csv",
        "text/csv"
    )
