# app.py  –  加入 CV / Range / Pdiff 特征版 + gap 差值模式展示（两列对齐表格 + 导出 + 强参考标记）
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
st.markdown("""
<style>
.stApp { background:linear-gradient(135deg,#f0f8ff,#e6e6fa); }
.company-name{font-size:1.1em;font-weight:600;text-shadow:1px 1px 2px rgba(0,0,0,0.2);}
.stTextInput>div>div>input{max-width:200px;}
.stButton>button{margin-top:4px;margin-bottom:8px;}

/* gap 差值模式表格：两列并排、行内对齐 */
.gap-table{
  width: auto;                          /* 表宽度按内容自适应 */
  margin: 0.5rem auto 0.8rem auto;      /* 居中 */
  border-collapse: separate;
  border-spacing: 15px 4px;             /* 两列间距 15px，行距 4px */
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

# ───────── 基本常量（云端用相对路径） ─────────
BASE_DIR  = Path(__file__).parent              # 当前 app.py 所在目录
DATA_FILE = BASE_DIR / "data" / "new_matches.xlsx"
HIST_PATH = BASE_DIR / "data" / "prediction_results (43).xlsx"

companies = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
outcomes  = ["主胜", "平局", "客胜"]
ODDS_COLS = [f"{c}_{o}" for c in companies for o in outcomes]  # 15 列

# ───────── 1. 衍生特征函数 ─────────
def add_extra_features(df_odds: pd.DataFrame) -> pd.DataFrame:
    """
    传入 15 列赔率 → 返回拼接 9 列新特征后的 DataFrame
    新特征:
        cv_home / cv_draw / cv_away
        range_home / range_draw / range_away
        p_diff (主/客平均隐含概率差)
    """
    extra = pd.DataFrame(index=df_odds.index)

    mat = df_odds.values.reshape(-1, len(companies), len(outcomes))  # (N,5,3)

    cv_vals    = np.std(mat, axis=1) / np.mean(mat, axis=1)
    range_vals = np.max(mat, axis=1) - np.min(mat, axis=1)
    extra[["cv_home", "cv_draw", "cv_away"]]           = cv_vals
    extra[["range_home", "range_draw", "range_away"]] = range_vals

    inv = 1 / mat
    imp = inv / inv.sum(axis=2, keepdims=True)
    p_diff = np.abs(imp[:, :, 0].mean(axis=1) - imp[:, :, 2].mean(axis=1))
    extra["p_diff"] = p_diff

    return pd.concat(
        [df_odds.reset_index(drop=True), extra.reset_index(drop=True)],
        axis=1
    )

# ───────── 1.1 gap 差值模式工具函数 ─────────
def _truncate_two_decimals(x: float) -> float:
    """
    只保留两位小数，不四舍五入。
    例如：0.3657 -> 0.36, 0.292 -> 0.29
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0

    if x >= 0:
        # 向下截断两位小数
        return math.floor(x * 100 + 1e-8) / 100.0
    else:
        # 若出现负数，用向上取整保证“截断”效果
        return math.ceil(x * 100 - 1e-8) / 100.0


def format_gap_pattern(values) -> str:
    """
    传入长度为 3 的 gap 列表，返回形如 '(3)7-4' 的字符串。
    1. 截断为两位小数；
    2. 升序排序；
    3. 计算相邻差值 *100 得到整数 d1, d2（不四舍五入）；
    4. 若有 0，则调整为 '0-X'；
    5. 外层为 |d1 - d2|。
    """
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
    """
    为指定 gap 列（如 'PRO_gap' 或 'PRO融合模型_gap'）计算
    0-1-2, 1-2-3, 2-3-4 三个窗口的差值模式。
    """
    if col not in sims.columns:
        return {}

    vals = sims[col].tolist()
    windows = [("0-1-2", 0), ("1-2-3", 1), ("2-3-4", 2)]
    patterns = {}

    for label, start in windows:
        if len(vals) >= start + 3:
            triple = vals[start:start + 3]
            patterns[label] = format_gap_pattern(triple)

    return patterns

# ───────── 2. 训练并缓存模型 ─────────
@st.cache_resource
def load_models():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")

    df_raw = pd.read_excel(DATA_FILE)
    X_base = df_raw[ODDS_COLS]
    X_feat = add_extra_features(X_base)          # 15+9 = 24 列
    feat_cols = X_feat.columns.tolist()

    # 隐含概率替换 (仅前 15 列赔率)
    X_imp = X_feat.copy().values.astype(float)
    for j in range(0, 15, 3):
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)

    y = df_raw["比赛结果"].map({"主胜": 0, "平局": 1, "客胜": 2}).values

    # —— Draw 二分类 —— #
    y_draw = (y == 1).astype(int)
    draw_hgb = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    ).fit(X_imp, y_draw)
    draw_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_imp, y_draw)

    # —— Win-Lose 二分类 —— #
    mask = (y != 1)
    X_wl, y_wl = X_imp[mask], (y[mask] == 0).astype(int)
    winlose_hgb = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    ).fit(X_wl, y_wl)

    # —— 备用多分类 (平局权重 0.8) —— #
    class_w = {0: 1.0, 1: 0.8, 2: 1.0}
    samp_w  = np.array([class_w[yi] for yi in y])
    multi_clf = HistGradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, loss="log_loss", random_state=42
    ).fit(X_imp, y, sample_weight=samp_w)

    return feat_cols, draw_hgb, draw_tree, winlose_hgb, multi_clf

feat_cols, draw_hgb, draw_tree, winlose_hgb, multi_clf = load_models()

# ───────── 3. Streamlit 页面 ─────────
st.set_page_config(page_title="CV/Range/Pdiff Boost 预测", layout="wide")
st.title("⚽足球预测系统")

# Session State
if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=ODDS_COLS)
if "matcher" not in st.session_state:
    if not HIST_PATH.exists():
        raise FileNotFoundError(f"HIST_PATH not found: {HIST_PATH}")
    st.session_state.matcher = SimilarityMatcher(str(HIST_PATH))
if "pattern_records" not in st.session_state:
    # 用于保存每场比赛的 6 个差值模式（导出用）
    st.session_state.pattern_records = []

# ---------- 数据输入 ----------
mode = st.radio("📥 数据输入方式", ["上传文件", "手动录入"], horizontal=True)

if mode == "上传文件":
    up = st.file_uploader("上传赔率文件 (Excel/CSV，每行15列)", type=["xlsx", "csv"])
    if up is not None:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.session_state.input_df = df_up[ODDS_COLS]
        st.session_state.pattern_records = []   # 输入变了，清空已有模式记录
        st.success(f"✅ 已读取 {len(df_up)} 场比赛")
        st.dataframe(st.session_state.input_df)

else:
    st.subheader("🖊 手动录入 (逐公司一行)")
    with st.form("manual", clear_on_submit=True):
        inps = {}
        for comp in companies:
            c1, c2 = st.columns([1, 2])
            c1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inps[comp] = c2.text_input(
                "", placeholder="2.05 3.60 3.50", key=f"man_{comp}"
            )
        if st.form_submit_button("添加比赛"):
            row, ok = [], True
            for comp in companies:
                parts = inps[comp].split()
                if len(parts) != 3:
                    st.error(f"{comp} 需输入 3 个赔率")
                    ok = False
                    break
                row += [float(x) for x in parts]
            if ok:
                st.session_state.input_df = pd.concat(
                    [st.session_state.input_df,
                     pd.DataFrame([row], columns=ODDS_COLS)],
                    ignore_index=True
                )
                st.session_state.pattern_records = []   # 有新比赛，清空旧模式记录
                st.success("✅ 已添加1场比赛")

# ---------- 历史匹配 ----------
if not st.session_state.input_df.empty:
    st.subheader("🔍 历史相似比赛推荐")
    df_pro    = predict_model_pro(st.session_state.input_df)
    prob_cols = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob_cols:
        df_pro[pc].fillna(0, inplace=True)

    ens_in = pd.concat(
        [
            st.session_state.input_df.reset_index(drop=True),
            df_pro[["average_gap"] + prob_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    try:
        df_ens = predict_model_pro_ensemble(ens_in)
    except Exception:
        df_ens = pd.DataFrame({
            "PRO融合模型预测结果": ["平局"] * len(df_pro),
            "PRO融合模型_gap": [0.0] * len(df_pro),
        })

    try:
        df_meta = predict_model_meta(st.session_state.input_df)
    except Exception:
        df_meta = pd.DataFrame()

    # 每次重新跑历史匹配前，清空 pattern_records，避免残留旧结果
    st.session_state.pattern_records = []

    # 模式列名（来自 gap_patterns_export.csv）
    pattern_cols_pro = [
        "PRO_pattern_0_1_2",
        "PRO_pattern_1_2_3",
        "PRO_pattern_2_3_4",
    ]
    pattern_cols_ens = [
        "ENS_pattern_0_1_2",
        "ENS_pattern_1_2_3",
        "ENS_pattern_2_3_4",
    ]

    for i in range(len(st.session_state.input_df)):
        # 1) 第一层：按 PRO_gap / PRO融合_gap / 融合信心 / 推荐总分 做相似匹配
        q = {
            "PRO_gap": df_pro.loc[i, "average_gap"],
            "PRO融合模型_gap": df_ens.loc[i, "PRO融合模型_gap"],
            "融合信心": df_meta.loc[i, "融合信心"] if "融合信心" in df_meta else 0,
            "推荐总分": df_meta.loc[i, "推荐总分"] if "推荐总分" in df_meta else 0,
            "pair": f"{df_pro.loc[i,'最终预测结果']}-{df_ens.loc[i,'PRO融合模型预测结果']}",
        }
        try:
            sims = st.session_state.matcher.query(q, k=5)
        except Exception:
            sims = pd.DataFrame()

        st.markdown(f"**第 {i+1} 场** 历史相似比赛：")

        # 保证推荐编号就是 0、1、2、3、4
        sims = sims.reset_index(drop=True)

        # ===== 计算当前比赛的 6 个模式（基于这 5 场相似比赛） =====
        pro_patterns = compute_gap_patterns(sims, "PRO_gap")
        ens_patterns = compute_gap_patterns(sims, "PRO融合模型_gap")

        pro0 = pro_patterns.get("0-1-2", "")
        pro1 = pro_patterns.get("1-2-3", "")
        pro2 = pro_patterns.get("2-3-4", "")

        ens0 = ens_patterns.get("0-1-2", "")
        ens1 = ens_patterns.get("1-2-3", "")
        ens2 = ens_patterns.get("2-3-4", "")

        # ===== 基于 6 个模式，打“强参考”标记 =====
        if not sims.empty and all(
            col in sims.columns for col in pattern_cols_pro + pattern_cols_ens
        ):
            # 处理缺失值为 ""，方便字符串对比
            for col in pattern_cols_pro + pattern_cols_ens:
                sims[col] = sims[col].fillna("")

            match_counts = []
            match_levels = []
            strong_flags  = []

            for _, row in sims.iterrows():
                cnt = 0
                if pro0 and row["PRO_pattern_0_1_2"] == pro0:
                    cnt += 1
                if pro1 and row["PRO_pattern_1_2_3"] == pro1:
                    cnt += 1
                if pro2 and row["PRO_pattern_2_3_4"] == pro2:
                    cnt += 1
                if ens0 and row["ENS_pattern_0_1_2"] == ens0:
                    cnt += 1
                if ens1 and row["ENS_pattern_1_2_3"] == ens1:
                    cnt += 1
                if ens2 and row["ENS_pattern_2_3_4"] == ens2:
                    cnt += 1

                match_counts.append(cnt)
                if cnt == 6:
                    level = "完全匹配"
                    strong = True
                elif cnt >= 4:
                    level = "基本匹配"
                    strong = True
                elif cnt >= 1:
                    level = "部分匹配"
                    strong = False
                else:
                    level = "不匹配"
                    strong = False

                match_levels.append(level)
                strong_flags.append(strong)

            sims["模式匹配个数"] = match_counts
            sims["模式匹配程度"] = match_levels
            sims["强参考"]      = strong_flags
            sims["强参考标记"]   = sims["强参考"].map(
                lambda x: "⭐ 强参考" if x else ""
            )
        else:
            sims["模式匹配个数"] = 0
            sims["模式匹配程度"] = "未提供模式"
            sims["强参考"]      = False
            sims["强参考标记"]   = ""

        # 展示相似比赛表（包含“强参考”列）
        st.dataframe(sims, use_container_width=True)

        # ===== 两个模型的差值模式，两列表格方式并排且纵向对齐（当前比赛自身的模式） =====
        if pro0 or pro1 or pro2 or ens0 or ens1 or ens2:
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

        # 保存导出记录：每场比赛 6 个模式
        st.session_state.pattern_records.append({
            "比赛编号": i + 1,
            "PRO_pattern_0_1_2": pro0,
            "PRO_pattern_1_2_3": pro1,
            "PRO_pattern_2_3_4": pro2,
            "ENS_pattern_0_1_2": ens0,
            "ENS_pattern_1_2_3": ens1,
            "ENS_pattern_2_3_4": ens2,
        })

    # —— 差值模式导出区域 —— #
    if st.session_state.pattern_records:
        df_patterns = pd.DataFrame(st.session_state.pattern_records).sort_values("比赛编号")
        st.subheader("📤 差值模式导出（每场比赛的 PRO / PRO融合 差值模式）")
        st.dataframe(df_patterns, use_container_width=True)
        st.download_button(
            "⬇️ 下载差值模式（CSV）",
            df_patterns.to_csv(index=False).encode("utf-8-sig"),
            "gap_patterns_export.csv",
            "text/csv",
        )

# ---------- 预测 ----------
if not st.session_state.input_df.empty and st.button("🎯 运行预测"):
    df_odds  = st.session_state.input_df.copy()
    X_feat   = add_extra_features(df_odds)                # 15+9 列
    X_imp    = X_feat.values.astype(float)
    for j in range(0, 15, 3):                             # 仅赔率列做隐含概率
        inv = 1 / X_imp[:, j:j+3]
        X_imp[:, j:j+3] = inv / inv.sum(axis=1, keepdims=True)

    # —— 1) Draw —— #
    p_draw = (
        0.6 * draw_hgb.predict_proba(X_imp)[:, 1]
        + 0.4 * draw_tree.predict_proba(X_imp)[:, 1]
    )
    p_draw = np.clip(p_draw + 0.10 * np.power(1 - p_draw, 0.50), 0, 1)

    # —— 2) Win-Lose —— #
    p_wl   = winlose_hgb.predict_proba(X_imp)
    p_base = np.zeros((len(X_imp), 3))
    p_base[:, 1] = p_draw
    p_base[:, 0] = p_wl[:, 1] * (1 - p_draw)
    p_base[:, 2] = p_wl[:, 0] * (1 - p_draw)

    # —— 3) PRO / Ensemble —— #
    df_pro  = predict_model_pro(df_odds)
    prob2   = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob2:
        df_pro[pc].fillna(0, inplace=True)

    ens_in  = pd.concat(
        [df_odds.reset_index(drop=True),
         df_pro[["average_gap"] + prob2].reset_index(drop=True)],
        axis=1,
    )
    try:
        df_ens = predict_model_pro_ensemble(ens_in)
        p_ens  = df_ens[[f"P({o})" for o in outcomes]].values
    except Exception:
        p_ens = np.zeros_like(p_base)

    # —— 4) META —— #
    try:
        df_meta = predict_model_meta(df_odds)
        p_meta  = df_meta[[f"P({o})" for o in outcomes]].values
    except Exception:
        p_meta = np.zeros_like(p_base)

    # —— 5) Multi —— #
    p_multi = multi_clf.predict_proba(X_imp)

    # —— 6) 五路融合 ——  ⬅︎ 用“四权重”替换均值
    w_base, w_ens, w_meta, w_multi = 0.10, 0.70, 0.10, 0.00
    wsum   = w_base + w_ens + w_meta + w_multi          # =1.0

    p_final = (
        w_base  * p_base
        + w_ens * p_ens
        + w_meta * p_meta
        + w_multi * p_multi
    ) / wsum

    p_final /= p_final.sum(axis=1, keepdims=True)        # 归一化

    preds = [outcomes[k] for k in p_final.argmax(axis=1)]
    df_res = pd.DataFrame(p_final * 100, columns=[f"{o}(%)" for o in outcomes])
    df_res.insert(0, "最终预测", preds)
    df_res.index = np.arange(1, len(df_res) + 1)
    df_res.index.name = "比赛编号"

    st.subheader("📊 综合模型预测结果")
    st.dataframe(df_res, use_container_width=True)
    st.download_button(
        "⬇️ 下载结果",
        df_res.to_csv(index=True).encode("utf-8-sig"),
        "predictions.csv",
        "text/csv",
    )
