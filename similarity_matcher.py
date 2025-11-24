"""
基于 PRO / PRO融合 等指标的历史相似比赛匹配器，并集成
gap 差值模式匹配（6 个模式字段）用于标记“强参考”比赛。

升级版：
- 仍保留“匹配个数”(0~6) 和 强参考 判断；
- 模式匹配规则改为：只看差值 Δ = |b - a|，差值相等即视为匹配成功
  例如：2-7 和 10-15 → Δ=5 和 Δ=5 → 计为匹配
- 在模式匹配场景（只传6个模式字段）下：
  - 先按匹配个数从多到少排序；
  - 若匹配个数相同，再按“模式距离”（平均 |Δ_new - Δ_hist|）从近到远排序。
- 新增：若历史库中不存在 'pair' 列，则自动用
    'PRO_最终预测结果' + '-' + 'PRO融合模型预测结果'
  或
    '最终预测结果' + '-' + 'PRO融合模型预测结果'
  构造 'pair' 列，以便 app.py 中的按 pair 过滤逻辑生效。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import re


def _parse_inner_pair(pat: str) -> Optional[Tuple[int, int]]:
    """
    从模式字符串中解析“最后两个整数”作为模式对。
    例如：
        "(3)7-4" -> (7, 4)
        "2-7"    -> (2, 7)
    若无法解析，返回 None。
    """
    if not isinstance(pat, str):
        return None
    nums = re.findall(r"-?\d+", pat)
    if len(nums) < 2:
        return None
    a, b = int(nums[-2]), int(nums[-1])
    return a, b


def _diff_from_pat(pat: str) -> Optional[float]:
    """
    从模式字符串中解析差值 Δ = |b - a|。
    例如：
        "(3)7-4" -> |4-7| = 3
        "2-7"    -> |7-2| = 5
    若无法解析，返回 None。
    """
    pair = _parse_inner_pair(pat)
    if pair is None:
        return None
    return abs(pair[1] - pair[0])


class SimilarityMatcher:
    def __init__(self, hist_path: str | Path, pattern_path: Optional[str | Path] = None):
        """
        hist_path : 关键历史结果文件路径（例如 prediction_results (43).xlsx）
        pattern_path : gap_patterns_export.csv 路径；若不传，则默认在同目录下寻找
                       文件名为 'gap_patterns_export.csv'
        """
        self.hist_path = Path(hist_path)

        # 1) 读取基础历史结果（xlsx 或 csv 都支持）
        if self.hist_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(self.hist_path)
        else:
            df = pd.read_csv(self.hist_path)

        df = df.copy()

        # 统一比赛编号字段名称：内部统一用 “比赛编号”
        if "比赛编号" not in df.columns:
            if "比赛序号" in df.columns:
                df.rename(columns={"比赛序号": "比赛编号"}, inplace=True)
            else:
                # 若原文件完全没有编号列，则用 index+1 自动生成
                df.insert(0, "比赛编号", np.arange(1, len(df) + 1, dtype=int))

        # 如果历史库中没有 'pair'，则自动构造：
        # 优先使用 PRO_最终预测结果 + PRO融合模型预测结果
        if "pair" not in df.columns:
            pro_col = None
            ens_col = None

            # 1) 优先尝试 'PRO_最终预测结果'
            if "PRO_最终预测结果" in df.columns:
                pro_col = "PRO_最终预测结果"
            elif "最终预测结果" in df.columns:
                pro_col = "最终预测结果"

            # 2) PRO融合模型预测结果
            if "PRO融合模型预测结果" in df.columns:
                ens_col = "PRO融合模型预测结果"

            if pro_col is not None and ens_col is not None:
                df["pair"] = df[pro_col].astype(str) + "-" + df[ens_col].astype(str)
            # 如果仍然构造不出来，就保持没有 pair 列，后续 self.pair_col 会是 None

        self.df = df

        # 2) 读取 gap 模式库，并 merge 进来
        if pattern_path is None:
            # 默认在同目录下找 gap_patterns_export.csv
            default_pattern = self.hist_path.with_name("gap_patterns_export.csv")
            pattern_path = default_pattern if default_pattern.exists() else None

        self.has_patterns = False
        self.pattern_cols_pro = [
            "PRO_pattern_0_1_2",
            "PRO_pattern_1_2_3",
            "PRO_pattern_2_3_4",
        ]
        self.pattern_cols_ens = [
            "ENS_pattern_0_1_2",
            "ENS_pattern_1_2_3",
            "ENS_pattern_2_3_4",
        ]

        if pattern_path is not None:
            p_path = Path(pattern_path)
            if p_path.exists():
                pat = pd.read_csv(p_path)

                # 同样统一为 “比赛编号”
                if "比赛编号" not in pat.columns:
                    if "比赛序号" in pat.columns:
                        pat.rename(columns={"比赛序号": "比赛编号"}, inplace=True)
                    else:
                        pat.insert(0, "比赛编号", np.arange(1, len(pat) + 1, dtype=int))

                # 按 “比赛编号” 左连接
                self.df = self.df.merge(pat, on="比赛编号", how="left")
                self.has_patterns = True

        # 3) 数值特征列 & 其他参考列
        self.numeric_cols = [
            c for c in ["PRO_gap", "PRO融合模型_gap", "融合信心", "推荐总分"]
            if c in self.df.columns
        ]
        # pair 列用于轻微惩罚 & app 中按 pair 过滤 Top5
        self.pair_col = "pair" if "pair" in self.df.columns else None

    # ───────────────────────── 内部工具：数值距离 ─────────────────────────
    def _compute_numeric_distance(self, q: Dict) -> np.ndarray:
        """
        仅基于数值型特征（PRO_gap / PRO融合模型_gap / 融合信心 / 推荐总分 / pair）
        计算欧氏距离的近似（平方和），值越小说明越相似。
        """
        n = len(self.df)
        dist = np.zeros(n, dtype=float)

        # 数值差异（平方和）
        for col in self.numeric_cols:
            if col in q and q[col] is not None:
                try:
                    qv = float(q[col])
                except Exception:
                    continue
                col_vals = self.df[col].astype(float)
                dist += (col_vals - qv) ** 2

        # pair 不同给轻微罚分（防止完全不同结果的比赛被排在前面）
        if self.pair_col is not None and "pair" in q and q["pair"] is not None:
            pair_penalty = (self.df[self.pair_col].astype(str) != str(q["pair"])).astype(
                float
            )
            dist += 0.1 * pair_penalty

        return dist

    # ───────────────────────── 内部工具：模式匹配 + 模式距离 ─────────────────────────
    def _compute_pattern_match(
        self, q: Dict
    ) -> tuple[np.ndarray, Optional[pd.Series], Optional[pd.Series], Optional[np.ndarray]]:
        """
        根据 query 中的 6 个模式字段，计算每条历史比赛与之的：
          - 匹配个数（基于差值 Δ = |b-a| 是否相等，0~6）
          - 匹配等级（完全/基本/部分/不匹配）
          - 强参考标记
          - 模式距离（平均 |Δ_new - Δ_hist|，越小越近）

        返回：
            match_counts : (N,) int，每条历史比赛匹配上的模式数量（0~6）
            match_level  : pd.Series 或 None
            strong_flag  : pd.Series 或 None
            pattern_dist : (N,) float 或 None（若未提供模式，则为 None）
        """
        if not self.has_patterns:
            n = len(self.df)
            return np.zeros(n, dtype=int), None, None, None

        # 从 query 中读取 6 个模式字符串，并转换为差值 Δ
        q_patterns_pro = [q.get(col) for col in self.pattern_cols_pro]
        q_patterns_ens = [q.get(col) for col in self.pattern_cols_ens]

        if all(p is None or p == "" for p in q_patterns_pro + q_patterns_ens):
            n = len(self.df)
            return np.zeros(n, dtype=int), None, None, None

        q_diffs_pro = [_diff_from_pat(p) if p not in (None, "") else None for p in q_patterns_pro]
        q_diffs_ens = [_diff_from_pat(p) if p not in (None, "") else None for p in q_patterns_ens]

        n = len(self.df)
        match_counts = np.zeros(n, dtype=int)

        # 用于计算“模式距离”（差值差的平均）
        sum_dist = np.zeros(n, dtype=float)
        cnt_used = np.zeros(n, dtype=int)

        # PRO 三个模式
        for idx, col in enumerate(self.pattern_cols_pro):
            qd = q_diffs_pro[idx]
            if qd is None or col not in self.df.columns:
                continue

            col_vals = self.df[col].astype(str).fillna("")
            for i_row, v in enumerate(col_vals):
                hd = _diff_from_pat(v)
                if hd is None:
                    continue
                # 匹配个数：差值 Δ 完全相同视为匹配成功
                if hd == qd:
                    match_counts[i_row] += 1
                # 距离：差值差的绝对值
                d = abs(qd - hd)
                sum_dist[i_row] += d
                cnt_used[i_row] += 1

        # ENS 三个模式
        for idx, col in enumerate(self.pattern_cols_ens):
            qd = q_diffs_ens[idx]
            if qd is None or col not in self.df.columns:
                continue

            col_vals = self.df[col].astype(str).fillna("")
            for i_row, v in enumerate(col_vals):
                hd = _diff_from_pat(v)
                if hd is None:
                    continue
                if hd == qd:
                    match_counts[i_row] += 1
                d = abs(qd - hd)
                sum_dist[i_row] += d
                cnt_used[i_row] += 1

        # 平均模式距离：没有任何可比字段的行，给一个很大的距离
        pattern_dist = np.full(n, 1e6, dtype=float)
        mask = cnt_used > 0
        pattern_dist[mask] = sum_dist[mask] / cnt_used[mask]

        # 匹配等级 + 强参考标记（仍基于匹配个数）
        levels = []
        strong_flags = []
        for c in match_counts:
            if c == 6:
                levels.append("完全匹配")
                strong_flags.append(True)
            elif c >= 4:  # 4 或 5 个匹配，视为“基本匹配”
                levels.append("基本匹配")
                strong_flags.append(True)
            elif c >= 1:
                levels.append("部分匹配")
                strong_flags.append(False)
            else:
                levels.append("不匹配")
                strong_flags.append(False)

        level_s = pd.Series(levels, index=self.df.index, name="模式匹配程度")
        strong_s = pd.Series(strong_flags, index=self.df.index, name="强参考")

        return match_counts, level_s, strong_s, pattern_dist

    # ───────────────────────── 对外接口 ─────────────────────────
    def query(self, q: Dict, k: Optional[int] = 5) -> pd.DataFrame:
        """
        根据 query 中的信息匹配最相似的历史比赛。

        两种使用场景：
        1）数值 + 可选模式：
            q 包含 PRO_gap / PRO融合模型_gap / 融合信心 / 推荐总分 / pair
            （这就是“历史相似比赛推荐”那块的用法）
            → 以数值距离为主，模式仅作为附加信息与标记。

        2）仅模式：
            q 仅包含 6 个模式字段：
                - "PRO_pattern_0_1_2"
                - "PRO_pattern_1_2_3"
                - "PRO_pattern_2_3_4"
                - "ENS_pattern_0_1_2"
                - "ENS_pattern_1_2_3"
                - "ENS_pattern_2_3_4"
            （这就是“模式匹配参考”那块的用法）
            → 只按模式匹配个数 + 模式距离排序：
                - 匹配个数多的排前；
                - 匹配个数一样时，模式差值越接近（avg |Δ_new-Δ_hist| 越小）的排前。
        """
        df = self.df.copy()

        # 判断是否用了数值特征
        numeric_used = any(col in q and q[col] is not None for col in self.numeric_cols) \
                       or ("pair" in q and q.get("pair") not in (None, ""))

        # 1) 数值距离
        numeric_dist = self._compute_numeric_distance(q) if numeric_used else np.zeros(len(df))

        # 2) 模式匹配 + 模式距离
        match_counts, match_level, strong_flag, pattern_dist = self._compute_pattern_match(q)

        # 3) 组合为总距离
        if numeric_used:
            # 场景 1：数值为主（历史相似比赛）
            dist = numeric_dist.copy()
            # 可选：匹配个数多略微减小距离
            if match_level is not None:
                dist = dist - 0.5 * match_counts
        else:
            # 场景 2：仅模式（模式匹配参考）
            if pattern_dist is None:
                dist = np.zeros(len(df))
            else:
                # 匹配个数优先（越多越好），其次模式距离越小越好
                dist = (6 - match_counts) * 1000.0 + pattern_dist

        df["_distance"] = dist

        # 附加模式匹配结果列
        if match_level is not None:
            df["模式匹配个数"] = match_counts
            df["模式匹配程度"] = match_level
            df["强参考"] = strong_flag
        else:
            df["模式匹配个数"] = 0
            df["模式匹配程度"] = "未提供模式"
            df["强参考"] = False

        # 排序 & 取前 k 条
        df_sorted = df.sort_values("_distance", ascending=True).reset_index(drop=True)
        if k is not None:
            df_sorted = df_sorted.head(k)

        return df_sorted
