# app.py  –  历史相似比赛 Top5 查看与导出
# + 历史 TOP5 的 6 个差值模式
# + 模式计数体系（equal_pro/diff1_pro/...）
# + PRO_gap & PRO融合模型_gap 等差递增三元组检测
#
# 显示内容（每场当前比赛）：
#   1. 历史相似 Top5（按 SimilarityMatcher.query 原始顺序）：
#       - 比赛序号
#       - 比赛结果
#       - PRO_最终预测结果
#       - PRO_gap（4 位小数）
#       - PRO融合模型预测结果
#       - PRO融合模型_gap（4 位小数）
#   2. 基于 Top5 计算的 6 个差值模式：
#       - PRO_gap：0-1-2, 1-2-3, 2-3-4
#       - PRO融合模型_gap：0-1-2, 1-2-3, 2-3-4
#   3. 基于 6 个模式的计数结果：
#       - equal_pro / diff1_pro
#       - equal_ens / diff1_ens
#       - equal_cross / diff1_cross
#       - total_count / parity(奇偶)
#   4. PRO_gap & PRO融合模型_gap 的等差递增三元组：
#       - 先截断 Top5 到两位小数（不四舍五入）
#       - 再判断是否存在任意三值构成严格递增的等差数列
#
# 导出内容（CSV）：
#   - 当前比赛序号、当前主队/客队
#   - 历史主队/客队
#   - 历史比赛序号、历史比赛结果、历史PRO_最终预测结果、历史PRO_gap、历史PRO融合模型预测结果、历史PRO融合模型_gap（4 位小数）
#   - equal_pro / diff1_pro / equal_ens / diff1_ens / equal_cross / diff1_cross / total_count / parity
#   - PRO_gap_top5_trunc / PRO_gap_has_ap / PRO_gap_ap_triplet
#   - ENS_gap_top5_trunc / ENS_gap_has_ap / ENS_gap_ap_triplet
#   - 在不同“当前比赛序号”的 Top5 之间插入一行空白行

import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

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

companies = ["Bet365","立博","Interwetten","Pinnacle","William Hill"]
TEAM_COLS = ["主队", "客队"]
outcomes  = ["主胜","平局","客胜"]
ODDS_COLS = [f"{c}_{o}" for c in companies for o in outcomes]
OUTCOME_COL = "比赛结果"

# ---------- 渲染紧凑表格 ----------
def render_compact_table(df: pd.DataFrame):
    """用固定列宽的 HTML 表格紧凑显示 DataFrame，并统一数值为 4 位小数。"""
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

# ---------- 差值模式工具 ----------
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
    """
    传入长度为 3 的 gap 列表，返回形如 '(outer)d1-d2' 的字符串。
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
    从历史相似比赛结果 sims 中，为指定 gap 列（如 'PRO_gap' 或 'PRO融合模型_gap'）
    计算 0-1-2, 1-2-3, 2-3-4 三个窗口的差值模式。
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

# ---------- PRO_gap & ENS_gap 等差三元组工具 ----------

def compute_gap_ap(sims: pd.DataFrame, col: str):
    """
    对当前比赛的 Top5某列 gap（如 PRO_gap 或 PRO融合模型_gap）：
      - 先截断到两位小数（不四舍五入）；
      - 再检查是否存在任意三个值构成严格递增的等差数列；
    返回：
      - trunc_list: 截断后的前 5 个 gap 列表（按 Top5 顺序）
      - has_ap: 1 / 0
      - ap_triplet: 若存在，返回 (a,b,c) 三个 float；否则 None
    """
    if col not in sims.columns or sims.empty:
        return [], 0, None
    gaps = sims[col].tolist()
    trunc = [_truncate_two_decimals(x) for x in gaps]
    ints = [int(round(v * 100)) for v in trunc]
    uniq = sorted(set(ints))

    has_ap = 0
    ap_triplet = None
    n = len(uniq)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = uniq[i], uniq[j], uniq[k]
                if b > a and c > b and (b - a) == (c - b):
                    has_ap = 1
                    ap_triplet = (a / 100.0, b / 100.0, c / 100.0)
                    break
            if has_ap:
                break
        if has_ap:
            break

    return trunc, has_ap, ap_triplet

# ---------- 模式解析 & 计数体系 ----------

def parse_pair_from_pattern(pat: str) -> Optional[Tuple[int,int]]:
    """
    从模式字符串中解析最后两个正整数 (a,b)：
      "(5)8-3" -> (8,3)
      "2-7"    -> (2,7)
    注意：忽略 '-' 符号，把它当分隔符，而不是负号。
    """
    if not isinstance(pat, str) or not pat.strip():
        return None
    nums = re.findall(r"\d+", pat)
    if len(nums) < 2:
        return None
    a, b = int(nums[-2]), int(nums[-1])
    return a, b

def delta_from_pattern(pat: str) -> Optional[int]:
    """Δ = |b - a|"""
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
    基于 6 个模式计算：
      - PRO 内部：equal_pro（Δ 相等）、diff1_pro（Δ 差1且两模式中至少有一个数字相同）
      - ENS 内部：equal_ens、diff1_ens
      - PRO vs ENS 交叉：equal_cross（Δ 相等）、diff1_cross（Δ 差1且两个模式之间无共同数字）
      - total_count / parity
    """

    pro_pats = [pro0, pro1, pro2]
    ens_pats = [ens0, ens1, ens2]

    pro_pairs = [parse_pair_from_pattern(p) for p in pro_pats]
    pro_deltas = [delta_from_pattern(p) for p in pro_pats]

    ens_pairs = [parse_pair_from_pattern(p) for p in ens_pats]
    ens_deltas = [delta_from_pattern(p) for p in ens_pats]

    # 1) PRO 内部
    equal_pro = 0
    diff1_pro = 0
    for i in range(3):
        for j in range(i+1, 3):
            di, dj = pro_deltas[i], pro_deltas[j]
            pi, pj = pro_pairs[i], pro_pairs[j]
            if di is None or dj is None or pi is None or pj is None:
                continue
            if di == dj:
                equal_pro += 1
            if abs(di - dj) == 1:
                if set(pi) & set(pj):
                    diff1_pro += 1

    # 2) ENS 内部
    equal_ens = 0
    diff1_ens = 0
    for i in range(3):
        for j in range(i+1, 3):
            di, dj = ens_deltas[i], ens_deltas[j]
            pi, pj = ens_pairs[i], ens_pairs[j]
            if di is None or dj is None or pi is None or pj is None:
                continue
            if di == dj:
                equal_ens += 1
            if abs(di - dj) == 1:
                if set(pi) & set(pj):
                    diff1_ens += 1

    # 3) PRO vs ENS 交叉：3×3 = 9 对
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
                # 两个模式之间不能有共同数字
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

# ───────── 3. Session 初始化 ─────────
st.title("⚽ 历史相似比赛 Top5 查看与导出")

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
    up = st.file_uploader("上传赔率文件 (建议包含主队/客队 + 15列赔率)", type=["xlsx", "csv"])
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
    st.subheader("🖊 手动录入 (逐公司一行)")
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
                    st.error(f"{comp} 需输入 3 个赔率"); ok = False; break
                try:
                    row_odds += [float(x) for x in parts]
                except ValueError:
                    st.error(f"{comp} 的赔率必须是数字"); ok = False; break
            if ok:
                new_row = pd.DataFrame([[home_team, away_team] + row_odds],
                                       columns=TEAM_COLS + ODDS_COLS)
                st.session_state.input_df = pd.concat(
                    [st.session_state.input_df, new_row],
                    ignore_index=True
                )
                st.success("✅ 已添加1场比赛")
                st.dataframe(st.session_state.input_df)

# ---------- 历史相似 Top5 显示 + 差值模式 + 计数 + 等差三元组 + 导出 ----------
if not st.session_state.input_df.empty and "matcher" in st.session_state:
    st.subheader("🔍 历史相似比赛 Top5（按 SimilarityMatcher 原始顺序）")

    df_odds = st.session_state.input_df[ODDS_COLS].copy()

    # 当前比赛 PRO 模型预测（用于 q_basic）
    df_pro = predict_model_pro(df_odds)
    prob_cols = [c for c in df_pro.columns if c.startswith("P(")]
    for pc in prob_cols:
        df_pro[pc].fillna(0, inplace=True)

    # PRO 融合输出
    ens_in = pd.concat([
        df_odds.reset_index(drop=True),
        df_pro[["average_gap"] + prob_cols].reset_index(drop=True)
    ], axis=1)
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

    matcher: SimilarityMatcher = st.session_state.matcher

    export_rows = []

    for i in range(len(st.session_state.input_df)):
        home = st.session_state.input_df.loc[i, "主队"] if "主队" in st.session_state.input_df.columns else ""
        away = st.session_state.input_df.loc[i, "客队"] if "客队" in st.session_state.input_df.columns else ""

        title_str = f"第 {i+1} 场"
        if home or away:
            title_str += f"：{home} vs {away}"
        st.markdown(f"### ▶ {title_str}")

        # 构造 q_basic
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

        # 严格使用 SimilarityMatcher 返回顺序：Top5
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

        # 构造用于显示的 DataFrame
        sims_show = sims_basic_full.copy()

        # 比赛序号：历史库中通常是“比赛编号”
        if "比赛序号" not in sims_show.columns:
            if "比赛编号" in sims_show.columns:
                sims_show["比赛序号"] = sims_show["比赛编号"]
            else:
                sims_show["比赛序号"] = ""

        # 比赛结果：自动从 比赛结果 / 比赛结果_y / 比赛结果_x 中取
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

        sims_show = sims_show[["比赛序号", "比赛结果", "PRO_最终预测结果", "PRO_gap",
                               "PRO融合模型预测结果", "PRO融合模型_gap"]]

        # 数值列保留 4 位小数（显示）
        for col in ["PRO_gap", "PRO融合模型_gap"]:
            if col in sims_show.columns:
                sims_show[col] = pd.to_numeric(sims_show[col], errors="coerce").round(4)

        # 显示 Top5 历史相似比赛
        render_compact_table(sims_show)

        # 显示历史 Top5 的 6 个差值模式
        if pro0 or pro1 or pro2 or ens0 or ens1 or ens2:
            st.markdown("**差值模式（基于历史 Top5）**")
            pattern_html = f"""
<table class="pattern-table">
  <tr>
    <th></th>
    <th>0-1-2</th>
    <th>1-2-3</th>
    <th>2-3-4</th>
  </tr>
  <tr>
    <th>PRO_gap</th>
    <td>{pro0}</td>
    <td>{pro1}</td>
    <td>{pro2}</td>
  </tr>
  <tr>
    <th>PRO融合模型_gap</th>
    <td>{ens0}</td>
    <td>{ens1}</td>
    <td>{ens2}</td>
  </tr>
</table>
"""
            st.markdown(pattern_html, unsafe_allow_html=True)

        # 计算当前比赛的模式计数
        counts = compute_pattern_counts_for_match(pro0, pro1, pro2, ens0, ens1, ens2)
        st.caption(
            f"计数结果："
            f"PRO 内 equal={counts['equal_pro']}, diff±1={counts['diff1_pro']}；"
            f"ENS 内 equal={counts['equal_ens']}, diff±1={counts['diff1_ens']}；"
            f"交叉 equal={counts['equal_cross']}, diff±1={counts['diff1_cross']}；"
            f"总次数={counts['total_count']}, parity={counts['parity']}（0=偶数,1=奇数）"
        )

        # 计算 PRO_gap Top5 截断值 + 等差三元组
        pro_trunc_list, pro_has_ap, pro_ap_triplet = compute_gap_ap(sims_basic_full, "PRO_gap")
        pro_trunc_str = ", ".join(f"{v:.2f}" for v in pro_trunc_list) if pro_trunc_list else "无"
        if pro_has_ap and pro_ap_triplet:
            pro_triplet_str = "、".join(f"{v:.2f}" for v in pro_ap_triplet)
            st.caption(
                f"PRO_gap Top5 截断两位小数后: {pro_trunc_str}；存在等差递增三元组: 是（{pro_triplet_str}）"
            )
        else:
            st.caption(
                f"PRO_gap Top5 截断两位小数后: {pro_trunc_str}；存在等差递增三元组: 否"
            )

        pro_trunc_str_for_export = ",".join(f"{v:.2f}" for v in pro_trunc_list)
        pro_ap_triplet_str = ""
        if pro_has_ap and pro_ap_triplet:
            pro_ap_triplet_str = "|".join(f"{v:.2f}" for v in pro_ap_triplet)

        # 计算 PRO融合模型_gap Top5 截断值 + 等差三元组
        ens_trunc_list, ens_has_ap, ens_ap_triplet = compute_gap_ap(sims_basic_full, "PRO融合模型_gap")
        ens_trunc_str = ", ".join(f"{v:.2f}" for v in ens_trunc_list) if ens_trunc_list else "无"
        if ens_has_ap and ens_ap_triplet:
            ens_triplet_str = "、".join(f"{v:.2f}" for v in ens_ap_triplet)
            st.caption(
                f"PRO融合模型_gap Top5 截断两位小数后: {ens_trunc_str}；存在等差递增三元组: 是（{ens_triplet_str}）"
            )
        else:
            st.caption(
                f"PRO融合模型_gap Top5 截断两位小数后: {ens_trunc_str}；存在等差递增三元组: 否"
            )

        ens_trunc_str_for_export = ",".join(f"{v:.2f}" for v in ens_trunc_list)
        ens_ap_triplet_str = ""
        if ens_has_ap and ens_ap_triplet:
            ens_ap_triplet_str = "|".join(f"{v:.2f}" for v in ens_ap_triplet)

        # ===== 准备导出行（保持当前循环的 Top5 顺序） =====
        hist_home_col = "主队" if "主队" in sims_basic_full.columns else None
        hist_away_col = "客队" if "客队" in sims_basic_full.columns else None

        for _, row in sims_basic_full.iterrows():
            export_rows.append({
                "当前比赛序号": i + 1,
                "当前主队": home,
                "当前客队": away,
                "历史比赛序号": row["比赛序号"] if "比赛序号" in row else row.get("比赛编号", ""),
                "历史比赛结果": get_result_value(row),
                "历史PRO_最终预测结果": row.get("PRO_最终预测结果", ""),
                "历史PRO_gap": row.get("PRO_gap", ""),
                "历史PRO融合模型预测结果": row.get("PRO融合模型预测结果", ""),
                "历史PRO融合模型_gap": row.get("PRO融合模型_gap", ""),
                "历史主队": row.get(hist_home_col, "") if hist_home_col else "",
                "历史客队": row.get(hist_away_col, "") if hist_away_col else "",
                "equal_pro": counts["equal_pro"],
                "diff1_pro": counts["diff1_pro"],
                "equal_ens": counts["equal_ens"],
                "diff1_ens": counts["diff1_ens"],
                "equal_cross": counts["equal_cross"],
                "diff1_cross": counts["diff1_cross"],
                "total_count": counts["total_count"],
                "parity": counts["parity"],
                "PRO_gap_top5_trunc": pro_trunc_str_for_export,
                "PRO_gap_has_ap": pro_has_ap,
                "PRO_gap_ap_triplet": pro_ap_triplet_str,
                "ENS_gap_top5_trunc": ens_trunc_str_for_export,
                "ENS_gap_has_ap": ens_has_ap,
                "ENS_gap_ap_triplet": ens_ap_triplet_str,
            })

    # 导出全部比赛的 Top5 历史相似列表
    if export_rows:
        df_export = pd.DataFrame(export_rows)

        # 数值列统一保留 4 位小数
        for col in ["历史PRO_gap", "历史PRO融合模型_gap"]:
            if col in df_export.columns:
                df_export[col] = pd.to_numeric(df_export[col], errors="coerce").round(4)

        # 不排序，保持 append 的 Top5 顺序；仅在比赛之间插入空行
        rows_with_blank = []
        last_match = None
        for _, row in df_export.iterrows():
            match_no = row["当前比赛序号"]
            if last_match is not None and match_no != last_match:
                rows_with_blank.append({col: "" for col in df_export.columns})
            rows_with_blank.append(row.to_dict())
            last_match = match_no

        df_export_with_blank = pd.DataFrame(rows_with_blank, columns=df_export.columns)

        st.subheader("📤 导出所有比赛的历史相似 Top5（含球队名称、模式计数与等差三元组）")
        render_compact_table(df_export_with_blank.head(30))  # 预览前 30 行

        st.download_button(
            "⬇️ 导出历史相似 Top5（CSV，4位小数+分组空行）",
            df_export_with_blank.to_csv(index=False).encode("utf-8-sig"),
            "all_matches_top5_history_with_counts_and_ap.csv",
            "text/csv",
        )
