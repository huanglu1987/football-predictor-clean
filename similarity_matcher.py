# similarity_matcher.py
"""
基于 PRO / PRO融合 等指标的历史相似比赛匹配器，并集成
gap 差值模式匹配（6 个模式字段）用于标记“强参考”比赛。

用法示例：
    from similarity_matcher import SimilarityMatcher

    matcher = SimilarityMatcher("data/prediction_results (43).xlsx")
    q = {
        "PRO_gap": 0.30,
        "PRO融合模型_gap": 0.25,
        "融合信心": 0.18,
        "推荐总分": 0.12,
        "pair": "主胜-主胜",
        # 可选：6 个模式字段（有则启用模式匹配逻辑）
        # "PRO_pattern_0_1_2": "(10)1-11",
        # ...
    }
    sims = matcher.query(q, k=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


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
        self.pair_col = "pair" if "pair" in self.df.columns else None

    # ───────────────────────── 内部工具 ─────────────────────────
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
            # 权重可以再微调，这里很小，只是轻微影响
            dist += 0.1 * pair_penalty

        return dist

    def _compute_pattern_match(
        self, q: Dict
    ) -> tuple[np.ndarray, Optional[pd.Series], Optional[pd.Series]]:
        """
        根据 query 中的 6 个模式字段，计算每条历史比赛与之的匹配个数和匹配等级。

        返回：
            match_counts : (N,) int，每条历史比赛匹配上的模式数量（0~6）
            match_level  : pd.Series 或 None，文字描述：完全匹配/基本匹配/部分匹配/不匹配
            strong_flag  : pd.Series 或 None，布尔型：是否可视为强参考（完全+基本）
        """
        if not self.has_patterns:
            n = len(self.df)
            return np.zeros(n, dtype=int), None, None

        # 从 query 中读取 6 个模式（可能不存在）
        q_patterns_pro = [q.get(col) for col in self.pattern_cols_pro]
        q_patterns_ens = [q.get(col) for col in self.pattern_cols_ens]

        # 如果一个都没提供，就不做模式匹配
        if all(p is None or p == "" for p in q_patterns_pro + q_patterns_ens):
            n = len(self.df)
            return np.zeros(n, dtype=int), None, None

        n = len(self.df)
        match_counts = np.zeros(n, dtype=int)

        # PRO 三个模式
        for idx, col in enumerate(self.pattern_cols_pro):
            qp = q_patterns_pro[idx]
            if qp is None or qp == "" or col not in self.df.columns:
                continue
            match_counts += (self.df[col].astype(str) == str(qp)).astype(int)

        # ENS 三个模式
        for idx, col in enumerate(self.pattern_cols_ens):
            qp = q_patterns_ens[idx]
            if qp is None or qp == "" or col not in self.df.columns:
                continue
            match_counts += (self.df[col].astype(str) == str(qp)).astype(int)

        # 文字等级 + 强参考标记
        levels = []
        strong_flags = []
        for c in match_counts:
            if c == 6:
                levels.append("完全匹配")
                strong_flags.append(True)
            elif c >= 4:  # 4 或 5 个匹配，视为“基本全部配对上”
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

        return match_counts, level_s, strong_s

    # ───────────────────────── 对外接口 ─────────────────────────
    def query(self, q: Dict, k: int = 5) -> pd.DataFrame:
        """
        根据 query 中的信息匹配最相似的历史比赛。

        q 至少应包含：
            - "PRO_gap"
            - "PRO融合模型_gap"
            - "融合信心"
            - "推荐总分"
            - "pair"
        可选再增加 6 个模式字段，用于模式匹配：
            - "PRO_pattern_0_1_2"
            - "PRO_pattern_1_2_3"
            - "PRO_pattern_2_3_4"
            - "ENS_pattern_0_1_2"
            - "ENS_pattern_1_2_3"
            - "ENS_pattern_2_3_4"

        返回：
            DataFrame，按综合距离升序排序，附加列：
                - _distance         : 数值特征距离（含少量模式加成）
                - 模式匹配个数      : 0~6（若有提供模式）
                - 模式匹配程度      : 完全匹配 / 基本匹配 / 部分匹配 / 不匹配
                - 强参考            : True / False
        """
        df = self.df.copy()

        # 1) 数值距离
        dist = self._compute_numeric_distance(q)

        # 2) 模式匹配（可选）
        match_counts, match_level, strong_flag = self._compute_pattern_match(q)

        # 将模式匹配轻微融入 distance：匹配越多 → distance 越小一点
        # （这里只做很轻微的“拉近”，主排序还是由数值特征决定）
        if match_level is not None:
            dist = dist - 0.5 * match_counts  # 系数可按需要再调

        df["_distance"] = dist

        # 附加模式匹配结果列
        if match_level is not None:
            df["模式匹配个数"] = match_counts
            df["模式匹配程度"] = match_level
            df["强参考"] = strong_flag
        else:
            # 若 query 未提供模式，则给出空列方便后续扩展
            df["模式匹配个数"] = 0
            df["模式匹配程度"] = "未提供模式"
            df["强参考"] = False

        # 按 distance 升序排序并取前 k 条
        df_sorted = df.sort_values("_distance", ascending=True).reset_index(drop=True)
        if k is not None:
            df_sorted = df_sorted.head(k)

        return df_sorted
