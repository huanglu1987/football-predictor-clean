# app.py  –  历史相似比赛 Top5 查看与导出
# + 历史 TOP5 的 6 个差值模式
# + 模式计数体系（equal_pro/diff1_pro/...）
# + PRO_gap & PRO融合模型_gap 等差递增三元组检测（截断两位，不四舍五入）
# + 5家公司主/平/客赔率：近似等差递增三元组（截断两位，不四舍五入；允许差值差<=0.01；列出具体三元组）
# + 新增：Top5 每行计算 gap_sum_100 = floor(PRO_gap*100) + floor(PRO融合模型_gap*100)
# + 新增：将“本次Top5序列”去匹配199库Top5序列，命中>=4/5则显示（顺序必须一致）

import math
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from models.pro_model import predict_model_pro
from models.model_pro_ensemble import predict_model_pro_ensemble
from models.predict_model_meta import predict_model_meta
from similarity_matcher import SimilarityMatcher

# ───────── UI 设置 ─────────
st.set_page_config(page_title="历史相似比赛 Top5 导出", layout="wide")

st.markdown("""
<style>
.stApp { background:linear-gradient(135deg,#f0f8ff,#e6e6fa); }
.company-name{font-size:1.05em;font-weight:600;text-shadow:1px 1px 2px rgba(0,0,0,0.2);}
.stTextInput>div>div>input{max-width:260px;}
.stButton>button{margin-top:4px;margin-bottom:8px;}

/* 紧凑历史表格样式 */
.hist-table{
  table-layout: fixed;
  width: 100%;
  border-collapse: collapse;
}
.hist-table th, .hist-table td{
  max-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 3px 4px;
  border-bottom: 1px solid #ddd;
  font-size: 0.85rem;
}
.hist-table th{
  font-weight: 600;
  background-color:#f5f5ff;
}

/* 差值模式的小表格 */
.pattern-table{
  border-collapse: collapse;
  margin-top: 4px;
}
.pattern-table th, .pattern-table td{
  border: 1px solid #ddd;
  padding: 3px 6px;
  font-size: 0.85rem;
}
.pattern-table th{
  background-color:#eef2ff;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ───────── 基本常量 ─────────
BASE_DIR  = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "new_matches.xlsx"
HIST_PATH = BASE_DIR / "data" / "prediction_results (43).xlsx"

# 199库Top5参考文件（相对路径，方便GitHub部署）
REF_TOP5_PATH = BASE_DIR / "data" / "all_matches_top5_history.csv"

companies = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
TEAM_COLS = ["主队", "客队"]
outcomes = ["主胜", "平局", "客胜"]
ODDS_COLS = [f"{c}_{o}" for c in companies for o in outcomes]
OUTCOME_COL = "比赛结果"

# ---------- 渲染紧凑表格 ----------
def render_compact_table(df: pd.DataFrame):
    """用固定列宽的 HTML 表格显示 DataFrame，并统一数值为 4 位小数。"""
    if df is None or df.empty:
        st.info("暂无数据。")
        return
    html = df.to_html(
        index=False,
        classes="hist-table",
        border=0,
        escape=False,
        float_format=lambda x: f"{x:.4f}",
    )
    st.markdown(html, unsafe_allow_html=True)

# ---------- 识别“比赛结果”列名 ----------
def get_result_value(row: pd.Series) -> str:
    """从一行中取比赛结果：优先比赛结果，其次比赛结果_y，再其次比赛结果_x。"""
    for col in ["比赛结果", "比赛结果_y", "比赛结果_x"]:
        if col in row and pd.notna(row[col]) and str(row[col]) != "":
            return str(row[col])
    return ""

# ---------- 截断工具：两位小数，不四舍五入 ----------
def _truncate_two_decimals(x: float) -> float:
    """截断到两位小数（不四舍五入）。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x >= 0:
        return math.floor(x * 100 + 1e-8) / 100.0
    else:
        return math.ceil(x * 100 - 1e-8) / 100.0

def _floor_times_100(x: float) -> int:
    """等价于：先截断到两位小数再×100（对正数就是 floor(x*100)）。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0
    if math.isnan(x):
        return 0
    return int(math.floor(x * 100 + 1e-8))

def compute_gap_sum_100(pro_gap: float, ens_gap: float) -> int:
    """
    1) 各取两位小数（不四舍五入）等价于 floor(x*100)/100
    2) 再×100等价于 floor(x*100)
    3) 两者相加
    """
    return _floor_times_100(pro_gap) + _floor_times_100(ens_gap)

# ---------- 差值模式工具 ----------
def format_gap_pattern(values) -> str:
    """3 个 gap → '(outer)d1-d2' 模式字符串。"""
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
    为指定 gap 列（如 'PRO_gap' 或 'PRO融合模型_gap'）计算：
      - 0-1-2, 1-2-3, 2-3-4 三个窗口的差值模式。
    """
    if col not in sims.columns:
        return {}
    vals = sims[col].tolist()
    patterns = {}
    for label, start in [("0-1-2", 0), ("1-2-3", 1), ("2-3-4", 2)]:
        if len(vals) >= start + 3:
            patterns[label] = format_gap_pattern(vals[start:start+3])
        else:
            patterns[label] = ""
    return patterns

# ---------- PRO_gap & ENS_gap 等差三元组工具 ----------
def compute_gap_ap(sims: pd.DataFrame, col: str):
    """
    对当前比赛的 Top5 某列 gap（如 PRO_gap 或 PRO融合模型_gap）：
      - 截断到两位小数（不四舍五入）；
      - 找严格递增等差三元组；
      - best_triplet 选公差最小的那组；
    """
    if col not in sims.columns or sims.empty:
        return [], 0, None, []

    gaps = sims[col].tolist()
    trunc = [_truncate_two_decimals(x) for x in gaps]

    ints = [int(round(v * 100)) for v in trunc]
    uniq = sorted(set(ints))

    has_ap = 0
    best_triplet = None
    best_step = None
    all_triplets: List[Tuple[float, float, float]] = []

    n = len(uniq)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = uniq[i], uniq[j], uniq[k]
                if b > a and c > b and (b - a) == (c - b):
                    has_ap = 1
                    step = b - a
                    triplet = (a / 100.0, b / 100.0, c / 100.0)
                    all_triplets.append(triplet)
                    if best_step is None or step < best_step:
                        best_step = step
                        best_triplet = triplet

    return trunc, has_ap, best_triplet, all_triplets

# ---------- 模式解析 & 计数体系 ----------
def parse_pair_from_pattern(pat: str) -> Optional[Tuple[int, int]]:
    if not isinstance(pat, str) or not pat.strip():
        return None
    nums = re.findall(r"\d+", pat)
    if len(nums) < 2:
        return None
    a, b = int(nums[-2]), int(nums[-1])
    return a, b

def delta_from_pattern(pat: str) -> Optional[int]:
    pair = parse_pair_from_pattern(pat)
    if pair is None:
        return None
    a, b = pair
    return abs(b - a)

def compute_pattern_counts_for_match(
    pro0: str, pro1: str, pro2: str,
    ens0: str, ens1: str, ens2: str,
):
    pro_pats = [pro0, pro1, pro2]
    ens_pats = [ens0, ens1, ens2]

    pro_pairs = [parse_pair_from_pattern(p) for p in pro_pats]
    pro_deltas = [delta_from_pattern(p) for p in pro_pats]

    ens_pairs = [parse_pair_from_pattern(p) for p in ens_pats]
    ens_deltas = [delta_from_pattern(p) for p in ens_pats]

    equal_pro = 0
    diff1_pro = 0
    for i in range(3):
        for j in range(i + 1, 3):
            di, dj = pro_deltas[i], pro_deltas[j]
            pi, pj = pro_pairs[i], pro_pairs[j]
            if di is None or dj is None or pi is None or pj is None:
                continue
            if di == dj:
                equal_pro += 1
            if abs(di - dj) == 1:
                if set(pi) & set(pj):
                    diff1_pro += 1

    equal_ens = 0
    diff1_ens = 0
    for i in range(3):
        for j in range(i + 1, 3):
            di, dj = ens_deltas[i], ens_deltas[j]
            pi, pj = ens_pairs[i], ens_pairs[j]
            if di is None or dj is None or pi is None or pj is None:
                continue
            if di == dj:
                equal_ens += 1
            if abs(di - dj) == 1:
                if set(pi) & set(pj):
                    diff1_ens += 1

    equal_cross = 0
    diff1_cross = 0
    for i in range(3):
        for j in range(3):
            di, dj = pro_deltas[i], ens_deltas[j]
            pi, pj = pro_pairs[i], ens_pairs[j]
            if di is None or dj is None or pi is None or pj is None:
                continue
            if di == dj:
                equal_cross += 1
            if abs(di - dj) == 1:
                if not (set(pi) & set(pj)):
                    diff1_cross += 1

    total = equal_pro + diff1_pro + equal_ens + diff1_ens + equal_cross + diff1_cross
    parity = 1 if total % 2 == 1 else 0

    return {
        "equal_pro": equal_pro,
        "diff1_pro": diff1_pro,
        "equal_ens": equal_ens,
        "diff1_ens": diff1_ens,
        "equal_cross": equal_cross,
        "diff1_cross": diff1_cross,
        "total_count": total,
        "parity": parity,
    }

# ========== 赔率（三元组近似等差）工具 ==========
def find_ap_triplets_for_odds(company_odds: List[Tuple[str, float]], tolerance_ticks: int = 1):
    """
    规则（近似等差）：
      - 赔率截断两位小数（不四舍五入）
      - 三元组按赔率从小到大
      - tick=0.01（两位小数*100）
      - |d1-d2|<=1 则近似等差
      - 严格递增
    """
    trunc_list = [(c, _truncate_two_decimals(v)) for c, v in company_odds]

    triplets_desc: List[str] = []
    for a in range(5):
        for b in range(a + 1, 5):
            for c in range(b + 1, 5):
                t = [trunc_list[a], trunc_list[b], trunc_list[c]]
                t_sorted = sorted(t, key=lambda x: x[1])

                v1, v2, v3 = t_sorted[0][1], t_sorted[1][1], t_sorted[2][1]
                if not (v1 < v2 < v3):
                    continue

                i1, i2, i3 = int(round(v1 * 100)), int(round(v2 * 100)), int(round(v3 * 100))
                d1_ticks = i2 - i1
                d2_ticks = i3 - i2

                if abs(d1_ticks - d2_ticks) <= tolerance_ticks:
                    d1 = d1_ticks / 100.0
                    d2 = d2_ticks / 100.0
                    delta = abs(d1_ticks - d2_ticks) / 100.0
                    triplets_desc.append(
                        f"{t_sorted[0][0]}:{v1:.2f} < {t_sorted[1][0]}:{v2:.2f} < {t_sorted[2][0]}:{v3:.2f}"
                        f" | d1={d1:.2f} d2={d2:.2f} |Δ|={delta:.2f}"
                    )

    trunc_vals = [v for _, v in trunc_list]
    return trunc_vals, triplets_desc

def analyze_odds_ap_for_match(input_row: pd.Series, tolerance_ticks: int = 1):
    result = {}
    for outcome in ["主胜", "平局", "客胜"]:
        company_odds = []
        for comp in companies:
            col = f"{comp}_{outcome}"
            company_odds.append((comp, float(input_row[col]) if pd.notna(input_row[col]) else 0.0))

        trunc_vals, trips = find_ap_triplets_for_odds(company_odds, tolerance_ticks=tolerance_ticks)
        key_prefix = {"主胜": "H", "平局": "D", "客胜": "A"}[outcome]

        result[f"{key_prefix}_odds_trunc"] = ",".join(f"{v:.2f}" for v in trunc_vals)
        result[f"{key_prefix}_odds_ap_has"] = 1 if trips else 0
        result[f"{key_prefix}_odds_ap_count"] = len(trips)
        result[f"{key_prefix}_odds_ap_triplets"] = "； ".join(trips)

    return result

# ========== 新增：199库Top5序列匹配 ==========
def _detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

@st.cache_data
def load_ref_top5_map(ref_path: Path) -> Dict[int, List[int]]:
    """
    读取199库Top5 CSV，生成：
      { 当前比赛序号(int): [历史比赛序号(int) * 5] }
    兼容列名：当前比赛序号/历史比赛序号（或比赛编号/比赛序号等）
    """
    if not ref_path.exists():
        return {}

    df = pd.read_csv(ref_path)

    cur_col = _detect_col(df, ["当前比赛序号", "当前比赛编号", "当前比赛ID", "当前序号"])
    hist_col = _detect_col(df, ["历史比赛序号", "历史比赛编号", "历史比赛ID", "历史序号"])

    # 兜底：如果没找到，尝试用“历史比赛序号”常见导出字段
    if cur_col == "" or hist_col == "":
        # 再兜底一次：有些导出可能叫“当前比赛序号”“历史比赛序号”，这里直接报错提示
        return {}

    df = df.dropna(subset=[cur_col, hist_col]).copy()
    df[cur_col] = pd.to_numeric(df[cur_col], errors="coerce")
    df[hist_col] = pd.to_numeric(df[hist_col], errors="coerce")
    df = df.dropna(subset=[cur_col, hist_col]).copy()
    df[cur_col] = df[cur_col].astype(int)
    df[hist_col] = df[hist_col].astype(int)

    ref_map: Dict[int, List[int]] = {}
    # 用行顺序累积，保证“前后顺序一致”
    for _, r in df.iterrows():
        k = int(r[cur_col])
        v = int(r[hist_col])
        ref_map.setdefault(k, []).append(v)

    # 只保留前5
    for k in list(ref_map.keys()):
        if len(ref_map[k]) >= 5:
            ref_map[k] = ref_map[k][:5]
        else:
            # 不足5的不参与匹配
            ref_map.pop(k, None)

    return ref_map

def match_top5_sequence(new_seq: List[int], ref_map: Dict[int, List[int]]) -> List[Dict[str, Any]]:
    """
    new_seq 长度=5。匹配规则：
      - 5/5：完全一致
      - 4/5：删掉1个元素后，存在4长子序列完全一致（顺序一致）
    返回匹配列表（level=5或4）
    """
    results: List[Dict[str, Any]] = []
    if len(new_seq) != 5 or not ref_map:
        return results

    new_tuple = tuple(new_seq)
    new_sub4 = [tuple([new_seq[j] for j in range(5) if j != drop_i]) for drop_i in range(5)]

    for ref_match_no, ref_seq in ref_map.items():
        if len(ref_seq) != 5:
            continue
        ref_tuple = tuple(ref_seq)

        if ref_tuple == new_tuple:
            results.append({
                "ref_match_no": ref_match_no,
                "level": 5,
                "ref_seq": ref_seq,
                "matched_subseq": None,
            })
            continue

        ref_sub4_set = set(tuple([ref_seq[k] for k in range(5) if k != drop_j]) for drop_j in range(5))
        hit_sub = None
        for sub in new_sub4:
            if sub in ref_sub4_set:
                hit_sub = list(sub)
                break

        if hit_sub is not None:
            results.append({
                "ref_match_no": ref_match_no,
                "level": 4,
                "ref_seq": ref_seq,
                "matched_subseq": hit_sub,
            })

    results.sort(key=lambda x: (-x["level"], x["ref_match_no"]))
    return results

# ───────── 页面标题 & Session 初始化 ─────────
st.title("⚽ 历史相似比赛 Top5 查看与导出（含 gap_sum_100 + 199库序列匹配）")

if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=TEAM_COLS + ODDS_COLS)

if "matcher" not in st.session_state:
    if not HIST_PATH.exists():
        st.error(f"找不到历史库文件：{HIST_PATH}")
    else:
        st.session_state.matcher = SimilarityMatcher(str(HIST_PATH))

# ---------- 数据输入 ----------
mode = st.radio("📥 数据输入方式", ["上传文件", "手动录入"], horizontal=True)

if mode == "上传文件":
    up = st.file_uploader("上传赔率文件（建议含主队/客队 + 15列赔率）", type=["xlsx", "csv"])
    if up is not None:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        for col in TEAM_COLS:
            if col not in df_up.columns:
                df_up[col] = ""
        missing_odds = [c for c in ODDS_COLS if c not in df_up.columns]
        if missing_odds:
            st.error(f"上传文件缺少以下赔率列：{missing_odds}")
        else:
            st.session_state.input_df = df_up[TEAM_COLS + ODDS_COLS].copy()
            st.success(f"✅ 已读取 {len(df_up)} 场比赛")
            st.dataframe(st.session_state.input_df)

else:
    st.subheader("🖊 手动录入（逐公司一行）")
    with st.form("manual", clear_on_submit=True):
        c1, c2 = st.columns(2)
        home_team = c1.text_input("主队名称", key="home_team")
        away_team = c2.text_input("客队名称", key="away_team")

        inps = {}
        st.markdown("请输入各博彩公司赔率（格式：主胜 平局 客胜，例如：2.05 3.60 3.50）")
        for comp in companies:
            r1, r2 = st.columns([1, 2])
            r1.markdown(f"<div class='company-name'>{comp}</div>", unsafe_allow_html=True)
            inps[comp] = r2.text_input("", placeholder="2.05 3.60 3.50", key=f"man_{comp}")

        if st.form_submit_button("添加比赛"):
            row_odds = []
            ok = True
            for comp in companies:
                parts = inps[comp].split()
                if len(parts) != 3:
                    st.error(f"{comp} 需输入 3 个赔率")
                    ok = False
                    break
                try:
                    row_odds += [float(x) for x in parts]
                except ValueError:
                    st.error(f"{comp} 的赔率必须是数字")
                    ok = False
                    break

            if ok:
                new_row = pd.DataFrame([[home_team, away_team] + row_odds],
                                       columns=TEAM_COLS + ODDS_COLS)
                st.session_state.input_df = pd.concat(
                    [st.session_state.input_df, new_row],
                    ignore_index=True
                )
                st.success("✅ 已添加1场比赛")
                st.dataframe(st.session_state.input_df)

# ---------- 主逻辑 ----------
if not st.session_state.input_df.empty and "matcher" in st.session_state:
    st.subheader("🔍 历史相似比赛 Top5（含 gap_sum_100 + 199库序列匹配）")

    matcher: SimilarityMatcher = st.session_state.matcher
    df_odds = st.session_state.input_df[ODDS_COLS].copy()

    # 当前比赛 PRO 模型输出
    df_pro = predict_model_pro(df_odds)
    prob_cols = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob_cols:
        df_pro[pc].fillna(0, inplace=True)

    # 融合输出
    ens_in = pd.concat(
        [df_odds.reset_index(drop=True),
         df_pro[["average_gap"] + prob_cols].reset_index(drop=True)],
        axis=1
    )
    try:
        df_ens = predict_model_pro_ensemble(ens_in)
    except Exception:
        df_ens = pd.DataFrame({
            "PRO融合模型预测结果": ["平局"] * len(df_pro),
            "PRO融合模型_gap": [0.0] * len(df_pro)
        })

    # META（用于 q_basic）
    try:
        df_meta = predict_model_meta(df_odds)
    except Exception:
        df_meta = pd.DataFrame()

    # 读取199库映射（用于序列匹配）
    ref_map_199 = load_ref_top5_map(REF_TOP5_PATH)
    if not ref_map_199:
        st.warning(f"⚠️ 199库Top5文件未加载成功或列名不匹配：{REF_TOP5_PATH}（需要包含：当前比赛序号、历史比赛序号）")

    export_rows = []

    for i in range(len(st.session_state.input_df)):
        home = st.session_state.input_df.loc[i, "主队"] if "主队" in st.session_state.input_df.columns else ""
        away = st.session_state.input_df.loc[i, "客队"] if "客队" in st.session_state.input_df.columns else ""

        title_str = f"第 {i+1} 场"
        if home or away:
            title_str += f"：{home} vs {away}"
        st.markdown(f"### ▶ {title_str}")

        # 赔率近似等差三元组（主/平/客，容差0.01）
        input_row = st.session_state.input_df.loc[i]
        odds_ap = analyze_odds_ap_for_match(input_row, tolerance_ticks=1)

        st.markdown("**赔率近似等差递增三元组（截断两位小数，不四舍五入；允许差值差≤0.01）**")
        st.caption(f"主胜5赔率(截断): {odds_ap['H_odds_trunc']} | 存在={odds_ap['H_odds_ap_has']} | 组数={odds_ap['H_odds_ap_count']}")
        if odds_ap["H_odds_ap_triplets"]:
            st.caption("主胜三元组： " + odds_ap["H_odds_ap_triplets"])

        st.caption(f"平局5赔率(截断): {odds_ap['D_odds_trunc']} | 存在={odds_ap['D_odds_ap_has']} | 组数={odds_ap['D_odds_ap_count']}")
        if odds_ap["D_odds_ap_triplets"]:
            st.caption("平局三元组： " + odds_ap["D_odds_ap_triplets"])

        st.caption(f"客胜5赔率(截断): {odds_ap['A_odds_trunc']} | 存在={odds_ap['A_odds_ap_has']} | 组数={odds_ap['A_odds_ap_count']}")
        if odds_ap["A_odds_ap_triplets"]:
            st.caption("客胜三元组： " + odds_ap["A_odds_ap_triplets"])

        # 历史相似 Top5
        curr_pro_res = df_pro.loc[i, "最终预测结果"] if "最终预测结果" in df_pro.columns else ""
        curr_ens_res = df_ens.loc[i, "PRO融合模型预测结果"] if "PRO融合模型预测结果" in df_ens.columns else ""
        curr_pair = f"{curr_pro_res}-{curr_ens_res}" if curr_pro_res and curr_ens_res else ""

        q_basic = {
            "PRO_gap": df_pro.loc[i, "average_gap"],
            "PRO融合模型_gap": df_ens.loc[i, "PRO融合模型_gap"],
            "融合信心": df_meta.loc[i, "融合信心"] if "融合信心" in df_meta.columns else 0,
            "推荐总分": df_meta.loc[i, "推荐总分"] if "推荐总分" in df_meta.columns else 0,
            "pair": curr_pair
        }

        try:
            sims_basic_full = matcher.query(q_basic, k=5)
        except Exception as e:
            st.warning(f"历史匹配调用出错：{e}")
            sims_basic_full = pd.DataFrame()

        sims_basic_full = sims_basic_full.reset_index(drop=True)
        if sims_basic_full.empty:
            st.info("未找到历史相似比赛。")
            continue

        # 计算 Top5 的 PRO / PRO融合 差值模式
        pro_patterns = compute_gap_patterns(sims_basic_full, "PRO_gap")
        ens_patterns = compute_gap_patterns(sims_basic_full, "PRO融合模型_gap")

        pro0 = pro_patterns.get("0-1-2", "")
        pro1 = pro_patterns.get("1-2-3", "")
        pro2 = pro_patterns.get("2-3-4", "")

        ens0 = ens_patterns.get("0-1-2", "")
        ens1 = ens_patterns.get("1-2-3", "")
        ens2 = ens_patterns.get("2-3-4", "")

        # ===== 新增：199库Top5序列匹配（>=4/5）=====
        # 获取当前Top5序列（优先比赛序号，否则比赛编号）
        id_col = "比赛序号" if "比赛序号" in sims_basic_full.columns else ("比赛编号" if "比赛编号" in sims_basic_full.columns else None)
        new_seq: List[int] = []
        if id_col is not None:
            new_seq = pd.to_numeric(sims_basic_full[id_col], errors="coerce").dropna().astype(int).tolist()[:5]

        if len(new_seq) == 5 and ref_map_199:
            matches_199 = match_top5_sequence(new_seq, ref_map_199)
            if matches_199:
                st.markdown("**🔁 199库 Top5 序列匹配命中（≥4/5 且顺序一致）**")
                show_df = pd.DataFrame([{
                    "命中等级": f"{m['level']}/5",
                    "参考库_当前比赛序号": m["ref_match_no"],
                    "参考库_Top5序列": "-".join(map(str, m["ref_seq"])),
                    "命中的4序列(若4/5)": "" if m["matched_subseq"] is None else "-".join(map(str, m["matched_subseq"]))
                } for m in matches_199])
                st.dataframe(show_df, use_container_width=True)
            else:
                st.caption("199库序列匹配：未命中 ≥4/5。")
        else:
            st.caption("199库序列匹配：当前Top5序列不足5个或199库未加载。")

        # 显示 Top5 表（新增 gap_sum_100 列）
        sims_show = sims_basic_full.copy()

        if "比赛序号" not in sims_show.columns:
            if "比赛编号" in sims_show.columns:
                sims_show["比赛序号"] = sims_show["比赛编号"]
            else:
                sims_show["比赛序号"] = ""

        if "比赛结果" not in sims_show.columns:
            if "比赛结果_y" in sims_show.columns:
                sims_show["比赛结果"] = sims_show["比赛结果_y"]
            elif "比赛结果_x" in sims_show.columns:
                sims_show["比赛结果"] = sims_show["比赛结果_x"]
            else:
                sims_show["比赛结果"] = ""

        for col in ["PRO_最终预测结果", "PRO_gap", "PRO融合模型预测结果", "PRO融合模型_gap"]:
            if col not in sims_show.columns:
                sims_show[col] = ""

        sims_show["gap_sum_100"] = sims_show.apply(
            lambda r: compute_gap_sum_100(r.get("PRO_gap", 0.0), r.get("PRO融合模型_gap", 0.0)),
            axis=1
        )

        sims_show = sims_show[[
            "比赛序号", "比赛结果",
            "PRO_最终预测结果", "PRO_gap",
            "PRO融合模型预测结果", "PRO融合模型_gap",
            "gap_sum_100"
        ]]

        sims_show["PRO_gap"] = pd.to_numeric(sims_show["PRO_gap"], errors="coerce").round(4)
        sims_show["PRO融合模型_gap"] = pd.to_numeric(sims_show["PRO融合模型_gap"], errors="coerce").round(4)

        render_compact_table(sims_show)

        # 模式表
        st.markdown("**差值模式（基于历史 Top5）**")
        st.markdown(f"""
<table class="pattern-table">
  <tr><th></th><th>0-1-2</th><th>1-2-3</th><th>2-3-4</th></tr>
  <tr><th>PRO_gap</th><td>{pro0}</td><td>{pro1}</td><td>{pro2}</td></tr>
  <tr><th>PRO融合模型_gap</th><td>{ens0}</td><td>{ens1}</td><td>{ens2}</td></tr>
</table>
""", unsafe_allow_html=True)

        # 模式计数
        counts = compute_pattern_counts_for_match(pro0, pro1, pro2, ens0, ens1, ens2)
        st.caption(
            f"计数结果：PRO equal={counts['equal_pro']}, diff±1={counts['diff1_pro']}；"
            f"ENS equal={counts['equal_ens']}, diff±1={counts['diff1_ens']}；"
            f"Cross equal={counts['equal_cross']}, diff±1={counts['diff1_cross']}；"
            f"Total={counts['total_count']}, parity={counts['parity']}"
        )

        # gap 等差三元组（PRO_gap）
        pro_trunc_list, pro_has_ap, pro_best_triplet, pro_all_triplets = compute_gap_ap(sims_basic_full, "PRO_gap")
        if pro_has_ap and pro_best_triplet:
            st.caption("PRO_gap 等差三元组(最小公差)： " + "、".join(f"{v:.2f}" for v in pro_best_triplet))
            if len(pro_all_triplets) > 1:
                st.caption("PRO_gap 全部等差三元组： " + "； ".join("、".join(f"{v:.2f}" for v in t) for t in pro_all_triplets))
        else:
            st.caption("PRO_gap：无等差三元组")

        # gap 等差三元组（PRO融合模型_gap）
        ens_trunc_list, ens_has_ap, ens_best_triplet, ens_all_triplets = compute_gap_ap(sims_basic_full, "PRO融合模型_gap")
        if ens_has_ap and ens_best_triplet:
            st.caption("PRO融合模型_gap 等差三元组(最小公差)： " + "、".join(f"{v:.2f}" for v in ens_best_triplet))
            if len(ens_all_triplets) > 1:
                st.caption("PRO融合模型_gap 全部等差三元组： " + "； ".join("、".join(f"{v:.2f}" for v in t) for t in ens_all_triplets))
        else:
            st.caption("PRO融合模型_gap：无等差三元组")

        # ===== 导出行（保持原逻辑）=====
        hist_home_col = "主队" if "主队" in sims_basic_full.columns else None
        hist_away_col = "客队" if "客队" in sims_basic_full.columns else None

        pro_best_triplet_str = "|".join(f"{v:.2f}" for v in pro_best_triplet) if pro_best_triplet else ""
        pro_all_triplets_str = ";".join("|".join(f"{v:.2f}" for v in t) for t in pro_all_triplets) if pro_all_triplets else ""

        ens_best_triplet_str = "|".join(f"{v:.2f}" for v in ens_best_triplet) if ens_best_triplet else ""
        ens_all_triplets_str = ";".join("|".join(f"{v:.2f}" for v in t) for t in ens_all_triplets) if ens_all_triplets else ""

        for _, r in sims_basic_full.iterrows():
            hist_pro_gap = r.get("PRO_gap", 0.0)
            hist_ens_gap = r.get("PRO融合模型_gap", 0.0)
            export_rows.append({
                "当前比赛序号": i + 1,
                "当前主队": home,
                "当前客队": away,

                "历史比赛序号": r.get("比赛序号", r.get("比赛编号", "")),
                "历史比赛结果": get_result_value(r),
                "历史PRO_最终预测结果": r.get("PRO_最终预测结果", ""),
                "历史PRO_gap": hist_pro_gap,
                "历史PRO融合模型预测结果": r.get("PRO融合模型预测结果", ""),
                "历史PRO融合模型_gap": hist_ens_gap,
                "历史gap_sum_100": compute_gap_sum_100(hist_pro_gap, hist_ens_gap),

                "历史主队": r.get(hist_home_col, "") if hist_home_col else "",
                "历史客队": r.get(hist_away_col, "") if hist_away_col else "",

                "equal_pro": counts["equal_pro"],
                "diff1_pro": counts["diff1_pro"],
                "equal_ens": counts["equal_ens"],
                "diff1_ens": counts["diff1_ens"],
                "equal_cross": counts["equal_cross"],
                "diff1_cross": counts["diff1_cross"],
                "total_count": counts["total_count"],
                "parity": counts["parity"],

                "PRO_gap_top5_trunc": ",".join(f"{v:.2f}" for v in pro_trunc_list),
                "PRO_gap_has_ap": pro_has_ap,
                "PRO_gap_ap_triplet": pro_best_triplet_str,
                "PRO_gap_ap_triplets_all": pro_all_triplets_str,

                "ENS_gap_top5_trunc": ",".join(f"{v:.2f}" for v in ens_trunc_list),
                "ENS_gap_has_ap": ens_has_ap,
                "ENS_gap_ap_triplet": ens_best_triplet_str,
                "ENS_gap_ap_triplets_all": ens_all_triplets_str,

                # 赔率近似等差（主/平/客）
                "H_odds_trunc": odds_ap["H_odds_trunc"],
                "H_odds_ap_has": odds_ap["H_odds_ap_has"],
                "H_odds_ap_count": odds_ap["H_odds_ap_count"],
                "H_odds_ap_triplets": odds_ap["H_odds_ap_triplets"],

                "D_odds_trunc": odds_ap["D_odds_trunc"],
                "D_odds_ap_has": odds_ap["D_odds_ap_has"],
                "D_odds_ap_count": odds_ap["D_odds_ap_count"],
                "D_odds_ap_triplets": odds_ap["D_odds_ap_triplets"],

                "A_odds_trunc": odds_ap["A_odds_trunc"],
                "A_odds_ap_has": odds_ap["A_odds_ap_has"],
                "A_odds_ap_count": odds_ap["A_odds_ap_count"],
                "A_odds_ap_triplets": odds_ap["A_odds_ap_triplets"],
            })

    # ===== 导出（插入空行分组）=====
    if export_rows:
        df_export = pd.DataFrame(export_rows)

        for col in ["历史PRO_gap", "历史PRO融合模型_gap"]:
            if col in df_export.columns:
                df_export[col] = pd.to_numeric(df_export[col], errors="coerce").round(4)

        rows_with_blank = []
        last_match = None
        for _, row in df_export.iterrows():
            match_no = row["当前比赛序号"]
            if last_match is not None and match_no != last_match:
                rows_with_blank.append({col: "" for col in df_export.columns})
            rows_with_blank.append(row.to_dict())
            last_match = match_no

        df_export_with_blank = pd.DataFrame(rows_with_blank, columns=df_export.columns)

        st.subheader("📤 导出所有比赛的历史相似 Top5（含 gap_sum_100）")
        render_compact_table(df_export_with_blank.head(20))

        st.download_button(
            "⬇️ 导出历史相似 Top5（CSV）",
            df_export_with_blank.to_csv(index=False).encode("utf-8-sig"),
            "all_matches_top5_history_with_gap_sum_100.csv",
            "text/csv",
        )
