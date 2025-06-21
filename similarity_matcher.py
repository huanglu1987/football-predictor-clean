# similarity_matcher.py
# ------------------------------------------------------------------
# 根据历史数据检索最相似比赛记录工具
# 支持部署到 GitHub：默认读取仓库根目录下 data/prediction_results (43).xlsx
# ------------------------------------------------------------------
import os
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 数值特征列
_NUM_COLS = ["PRO_gap", "PRO融合模型_gap", "融合信心", "推荐总分"]
# 特征权重与配对惩罚
_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.6])
_PAIR_PENALTY = 5.0

class SimilarityMatcher:
    def __init__(self, hist_path: Optional[str] = None):
        """
        hist_path: 历史数据 Excel 路径，默认 data/prediction_results (43).xlsx
        """
        if hist_path is None:
            # 默认使用相对路径 data/ 目录
            base = os.path.dirname(os.path.abspath(__file__))
            hist_path = os.path.join(base, "data", "prediction_results (43).xlsx")
        # 读取原始历史数据
        self.orig_df = pd.read_excel(hist_path)
        # 归一化副本
        self.df_norm = self.orig_df.copy()
        # 构建配对标签
        self.df_norm['pair'] = (
            self.df_norm['PRO_最终预测结果'] + '-' +
            self.df_norm['PRO融合模型预测结果']
        )
        # 标准化数值特征
        self.scaler = StandardScaler().fit(self.df_norm[_NUM_COLS])
        self.df_norm[_NUM_COLS] = self.scaler.transform(self.df_norm[_NUM_COLS])

    def _distance(self, q_vec: np.ndarray, q_pair: str) -> np.ndarray:
        """计算查询向量到所有历史记录的距离"""
        diff = self.df_norm[_NUM_COLS].values - q_vec
        # 加权欧氏距离
        dists = np.sqrt((diff ** 2 * _WEIGHTS).sum(axis=1))
        # 若配对不同，则加惩罚
        dists += np.where(self.df_norm['pair'] == q_pair, 0, _PAIR_PENALTY)
        return dists

    def query(self, row: dict, k: int = 5) -> pd.DataFrame:
        """
        row: 包含 _NUM_COLS + 'pair' 字段的字典
        k: 返回最相似比赛数量
        返回值：原始历史记录（DataFrame）+ '_distance' 列
        """
        # 准备查询向量
        q_vals = [row[col] for col in _NUM_COLS]
        q_vec = self.scaler.transform([q_vals])[0]
        q_pair = row['pair']
        # 计算距离并排序
        dists = self._distance(q_vec, q_pair)
        idx = np.argsort(dists)[:k]
        # 返回原始记录 + 距离
        result = self.orig_df.iloc[idx].copy().reset_index(drop=True)
        result['_distance'] = dists[idx]
        return result
