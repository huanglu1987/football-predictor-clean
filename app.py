# app.py  –  历史相似比赛 Top5 查看与导出
# + 历史 TOP5 的 6 个差值模式
# + 模式计数体系（equal_pro/diff1_pro/...）
# + PRO_gap & PRO融合模型_gap 等差递增三元组检测（截断两位，不四舍五入）
# + 5家公司主/平/客赔率：近似等差递增三元组（截断两位，不四舍五入；允许差值差<=0.01；列出具体三元组）
# + Top5 每行计算 gap_sum_100 = floor(PRO_gap*100) + floor(PRO融合模型_gap*100)
# + 199库 Top5 序列匹配（>=4/5 且顺序一致），并显示“当前比赛结果”
# + 保留：导出历史相似TOP5（CSV，分组空行）
# + 新增：Top3（Top5第3行）规则提示（9条）
#   - Parity=0/1：由“整体 total_count”的奇偶得到（偶0奇1）
#   - “Parity=3”：实际指整体 total_count==3
#   - 页面展示 Parity 用 total_count（不展示0/1）

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
  max-width: 120px;
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

# 199库（未来扩展到300也一样）相对路径
REF_TOP5_PATH = BASE_DIR / "data" / "all_matches_top5_history.csv"

companies = ["Bet365", "立博", "Interwetten", "Pinnacle", "William Hill"]
TEAM_COLS = ["主队", "客队"]
outcomes = ["主胜", "平局", "客胜"]
ODDS_COLS = [f"{c}_{o}" for c in companies for o in outcomes]

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
    """对正数等价于 floor(x*100)。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0
    if pd.isna(x):
        return 0
    return int(math.floor(x * 100 + 1e-8))

def compute_gap_sum_100(pro_gap: float, ens_gap: float) -> int:
    """floor(PRO_gap*100) + floor(ENS_gap*100)"""
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
    """为指定 gap 列计算 0-1-2 / 1-2-3 / 2-3-4 三个窗口的差值模式。"""
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
    """Top5 gap 截断两位后，找严格递增等差三元组；best 取公差最小的一组。"""
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

# ---------- 模式解析 & 计数体系（返回 total_count + parity_bit） ----------
def parse_pair_from_pattern(pat: str) -> Optional[Tuple[int, int]]:
    if not isinstance(pat, str) or not pat.strip():
        return None
    nums = re.findall(r"\d+", pat)
    if len(nums) < 2:
        return None
    return int(nums[-2]), int(nums[-1])

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
    """
    返回：
      total_count = 计数总次数（整数）
      parity_bit = total_count奇偶（偶0奇1）
    """
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
            if abs(di - dj) == 1 and (set(pi) & set(pj)):
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
            if abs(di - dj) == 1 and (set(pi) & set(pj)):
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
            if abs(di - dj) == 1 and not (set(pi) & set(pj)):
                diff1_cross += 1

    total = equal_pro + diff1_pro + equal_ens + diff1_ens + equal_cross + diff1_cross
    parity_bit = 1 if (total % 2) == 1 else 0

    return {
        "equal_pro": equal_pro,
        "diff1_pro": diff1_pro,
        "equal_ens": equal_ens,
        "diff1_ens": diff1_ens,
        "equal_cross": equal_cross,
        "diff1_cross": diff1_cross,
        "total_count": total,
        "parity_bit": parity_bit,
    }

# ========== 赔率（三元组近似等差）工具 ==========
def find_ap_triplets_for_odds(company_odds: List[Tuple[str, float]], tolerance_ticks: int = 1):
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

# ========== 参考库 Top5序列匹配（带“当前比赛结果”） ==========
def _detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

@st.cache_data
def load_ref_top5_map_with_result(ref_path: Path) -> Dict[int, Dict[str, Any]]:
    if not ref_path.exists():
        return {}

    df = pd.read_csv(ref_path)
    cur_col = _detect_col(df, ["当前比赛序号"])
    hist_col = _detect_col(df, ["历史比赛序号"])
    res_col = _detect_col(df, ["当前比赛结果"])

    if cur_col == "" or hist_col == "":
        return {}

    df = df.dropna(subset=[cur_col, hist_col]).copy()
    df[cur_col] = pd.to_numeric(df[cur_col], errors="coerce")
    df[hist_col] = pd.to_numeric(df[hist_col], errors="coerce")
    df = df.dropna(subset=[cur_col, hist_col]).copy()
    df[cur_col] = df[cur_col].astype(int)
    df[hist_col] = df[hist_col].astype(int)

    ref_map: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        k = int(r[cur_col])
        v = int(r[hist_col])

        if k not in ref_map:
            ref_map[k] = {"seq": [], "result": ""}

        if len(ref_map[k]["seq"]) < 5:
            ref_map[k]["seq"].append(v)

        if res_col != "" and ref_map[k]["result"] == "":
            val = r.get(res_col, "")
            if pd.notna(val) and str(val).strip() != "":
                ref_map[k]["result"] = str(val).strip()

    for k in list(ref_map.keys()):
        if len(ref_map[k]["seq"]) != 5:
            ref_map.pop(k, None)

    return ref_map

def match_top5_sequence(new_seq: List[int], ref_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if len(new_seq) != 5 or not ref_map:
        return results

    new_tuple = tuple(new_seq)
    new_sub4 = [tuple([new_seq[j] for j in range(5) if j != drop_i]) for drop_i in range(5)]

    for ref_match_no, obj in ref_map.items():
        ref_seq = obj.get("seq", [])
        ref_res = obj.get("result", "")
        if len(ref_seq) != 5:
            continue

        ref_tuple = tuple(ref_seq)
        if ref_tuple == new_tuple:
            results.append({
                "ref_match_no": ref_match_no,
                "level": 5,
                "ref_seq": ref_seq,
                "ref_result": ref_res,
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
                "ref_result": ref_res,
                "matched_subseq": hit_sub,
            })

    results.sort(key=lambda x: (-x["level"], x["ref_match_no"]))
    return results

# ========== Top3 规则提示（使用“整体 total_count/parity_bit”，不是Top3行的Parity） ==========
def _in_range(x: Optional[float], lo: float, hi: float) -> bool:
    if x is None or pd.isna(x):
        return False
    return (x >= lo) and (x <= hi)

def check_top3_rules(
    sims_basic_full: pd.DataFrame,
    overall_total_count: Optional[int],
    overall_parity_bit: Optional[int],
) -> List[str]:
    """
    基于Top3（Top5第3行）读取模型预测与ens_gap，
    但 Parity=0/1/3 全部来自“整体 total_count/parity_bit”：
      - Parity=0/1：overall_parity_bit（偶0奇1）
      - Parity=3：overall_total_count==3
    """
    msgs: List[str] = []
    if sims_basic_full is None or sims_basic_full.empty or len(sims_basic_full) < 3:
        return msgs

    top3 = sims_basic_full.iloc[2]

    pro_pred = str(top3.get("PRO_最终预测结果", "")).strip()
    ens_pred = str(top3.get("PRO融合模型预测结果", "")).strip()

    try:
        ens_gap = float(top3.get("PRO融合模型_gap", np.nan))
    except Exception:
        ens_gap = np.nan

    agree = (pro_pred != "") and (pro_pred == ens_pred)
    ens_is_away = (ens_pred == "客胜")
    ens_is_home = (ens_pred == "主胜")

    # 1) Top3：融合客胜 + ens_gap 0.05-0.1 + Parity=0（整体偶）
    if ens_is_away and _in_range(ens_gap, 0.05, 0.10) and overall_parity_bit == 0:
        msgs.append(f"规则1：Top3 融合=客胜 & 融合gap∈[0.05,0.10] & total_count为偶数（total_count={overall_total_count}）")

    # 2) Top3：融合客胜 + ens_gap 0.15-0.2 + Parity=1（整体奇）
    if ens_is_away and _in_range(ens_gap, 0.15, 0.20) and overall_parity_bit == 1:
        msgs.append(f"规则2：Top3 融合=客胜 & 融合gap∈[0.15,0.20] & total_count为奇数（total_count={overall_total_count}）")

    # 3) Top3：融合客胜 + ens_gap 0.05-0.1（不看Parity）
    if ens_is_away and _in_range(ens_gap, 0.05, 0.10):
        msgs.append("规则3：Top3 融合=客胜 & 融合gap∈[0.05,0.10]（不看Parity）")

    # 4) Top3：融合主胜 + Parity=3（整体 total_count==3）
    if ens_is_home and (overall_total_count == 3):
        msgs.append("规则4：Top3 融合=主胜 & total_count==3")

    # 5) Top3：两模型一致+预测客胜+ens_gap 0.05-0.1
    if agree and ens_is_away and _in_range(ens_gap, 0.05, 0.10):
        msgs.append("规则5：Top3 两模型一致=客胜 & 融合gap∈[0.05,0.10]")

    # 6) Top3：融合主胜+ens_gap 0-0.02 + total_count==3
    if ens_is_home and _in_range(ens_gap, 0.00, 0.02) and (overall_total_count == 3):
        msgs.append("规则6：Top3 融合=主胜 & 融合gap∈[0.00,0.02] & total_count==3")

    # 7) Top3：PRO=平局 + 融合=主/客 + ens_gap 0.02-0.05
    if (pro_pred == "平局") and (ens_pred in ["主胜", "客胜"]) and _in_range(ens_gap, 0.02, 0.05):
        msgs.append("规则7：Top3 PRO=平局 & 融合=主胜/客胜 & 融合gap∈[0.02,0.05]")

    # 8) Top3：两模型一致+预测客胜+ens_gap 0.15-0.2
    if agree and ens_is_away and _in_range(ens_gap, 0.15, 0.20):
        msgs.append("规则8：Top3 两模型一致=客胜 & 融合gap∈[0.15,0.20]")

    # 9) Parity=3：整体 total_count==3
    if overall_total_count == 3:
        msgs.append("规则9：整体 total_count==3（独立提示）")

    # 去重
    uniq = []
    for m in msgs:
        if m not in uniq:
            uniq.append(m)
    return uniq


# ───────── 页面标题 & Session 初始化 ─────────
st.title("⚽ 历史相似比赛 Top5 查看与导出（Parity展示为 total_count）")

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
    st.subheader("🔍 历史相似比赛 Top5（含 total_count & Top3提示）")

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

    # 读取参考库映射（含当前比赛结果）
    ref_map = load_ref_top5_map_with_result(REF_TOP5_PATH)
    if not ref_map:
        st.warning(f"⚠️ 参考库未加载成功或列名不匹配：{REF_TOP5_PATH}（至少需要：当前比赛序号、历史比赛序号；可选：当前比赛结果）")

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

        # ===== 先计算“整体 total_count/parity_bit”（来自模式计数体系）=====
        # 先从Top5计算两套 gap 的差值模式
        pro_patterns = compute_gap_patterns(sims_basic_full, "PRO_gap")
        ens_patterns = compute_gap_patterns(sims_basic_full, "PRO融合模型_gap")
        pro0, pro1, pro2 = pro_patterns.get("0-1-2", ""), pro_patterns.get("1-2-3", ""), pro_patterns.get("2-3-4", "")
        ens0, ens1, ens2 = ens_patterns.get("0-1-2", ""), ens_patterns.get("1-2-3", ""), ens_patterns.get("2-3-4", "")

        counts = compute_pattern_counts_for_match(pro0, pro1, pro2, ens0, ens1, ens2)
        overall_total_count = counts["total_count"]
        overall_parity_bit = counts["parity_bit"]

        # 展示 Parity（按你的要求：展示总次数，不显示0/1）
        st.caption(f"**整体计数总次数 total_count = {overall_total_count}**（奇偶用于规则判断：{'奇' if overall_parity_bit==1 else '偶'}）")

        # ===== Top3规则提示：Parity使用整体total_count =====
        top3_msgs = check_top3_rules(
            sims_basic_full=sims_basic_full,
            overall_total_count=overall_total_count,
            overall_parity_bit=overall_parity_bit,
        )
        if top3_msgs:
            st.warning("⚠️ Top3 触发规则提示：\n\n- " + "\n- ".join(top3_msgs))

        # 当前Top5序列（比赛序号 or 比赛编号）
        id_col = "比赛序号" if "比赛序号" in sims_basic_full.columns else ("比赛编号" if "比赛编号" in sims_basic_full.columns else None)
        new_seq: List[int] = []
        if id_col is not None:
            new_seq = pd.to_numeric(sims_basic_full[id_col], errors="coerce").dropna().astype(int).tolist()[:5]

        # 参考库序列匹配（≥4/5）
        if len(new_seq) == 5 and ref_map:
            matches = match_top5_sequence(new_seq, ref_map)
            if matches:
                st.markdown("**🔁 参考库 Top5 序列匹配命中（≥4/5 且顺序一致）**")
                show_df = pd.DataFrame([{
                    "命中等级": f"{m['level']}/5",
                    "参考库_当前比赛序号": m["ref_match_no"],
                    "当前比赛结果": m.get("ref_result", ""),
                    "参考库_Top5序列": "-".join(map(str, m["ref_seq"])),
                    "命中的4序列(若4/5)": "" if m["matched_subseq"] is None else "-".join(map(str, m["matched_subseq"])),
                } for m in matches])
                st.dataframe(show_df, use_container_width=True)
            else:
                st.caption("参考库序列匹配：未命中 ≥4/5。")
        else:
            st.caption("参考库序列匹配：当前Top5序列不足5个或参考库未加载。")

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

        # 你要展示 total_count（整体），这里加一列给Top5每行都同样值（便于导出/查看）
        sims_show["total_count"] = overall_total_count

        sims_show = sims_show[[
            "比赛序号", "比赛结果",
            "PRO_最终预测结果", "PRO_gap",
            "PRO融合模型预测结果", "PRO融合模型_gap",
            "gap_sum_100",
            "total_count"
        ]]

        sims_show["PRO_gap"] = pd.to_numeric(sims_show["PRO_gap"], errors="coerce").round(4)
        sims_show["PRO融合模型_gap"] = pd.to_numeric(sims_show["PRO融合模型_gap"], errors="coerce").round(4)

        render_compact_table(sims_show)

        # ===== 导出Top5（CSV，分组空行）=====
        hist_home_col = "主队" if "主队" in sims_basic_full.columns else None
        hist_away_col = "客队" if "客队" in sims_basic_full.columns else None

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

                # 新增：整体 total_count（作为你要展示的“Parity总次数”）
                "total_count": overall_total_count,

                "历史主队": r.get(hist_home_col, "") if hist_home_col else "",
                "历史客队": r.get(hist_away_col, "") if hist_away_col else "",
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

        st.subheader("📤 导出所有比赛的历史相似 Top5（CSV，分组空行）")
        render_compact_table(df_export_with_blank.head(20))

        st.download_button(
            "⬇️ 导出历史相似 Top5（CSV）",
            df_export_with_blank.to_csv(index=False).encode("utf-8-sig"),
            "all_matches_top5_history_export.csv",
            "text/csv",
        )
