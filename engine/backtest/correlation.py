"""6.1 节"相关性分析反推权重" + 6.3 节"样本量门槛"：
compute_information_coefficient()。

信息系数（IC）= 某个子指标的历史取值 与 对应的未来实际收益 之间的相关
系数。IC 越高代表这个子指标越能预测未来收益，是"数据驱动权重"的具体
计算依据（M3 阶段拿这个结果去调整第 5 章的 V1 权重表）。

输出字段对齐 `data/schema.sql` 的 `indicator_correlation_history` 表：
    sample_size, correlation_ic, is_reliable, suggested_weight
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

#: 6.3 节硬性要求："任何子指标/场景的信号数量低于 30 次之前，胜率或相关
#: 系数标注为'样本不足，暂不采信'"。
MIN_RELIABLE_SAMPLE_SIZE = 30


@dataclass
class ICResult:
    """一次 IC 计算的结果，字段对齐 `indicator_correlation_history` 表。"""

    sample_size: int
    correlation_ic: Optional[float]
    is_reliable: bool
    indicator_name: Optional[str] = None
    scenario: Optional[str] = None
    # ⚠️ 建议权重的具体计算公式（IC 怎么换算成权重）规格书没有定义，属于
    # 需要单独设计、需要你确认的另一个问题，本函数不擅自发明一个公式，
    # 先留空占位，等 M3 阶段真的要做权重迭代时再实现。
    suggested_weight: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "indicator_name": self.indicator_name,
            "scenario": self.scenario,
            "sample_size": self.sample_size,
            "correlation_ic": self.correlation_ic,
            "suggested_weight": self.suggested_weight,
            "is_reliable": self.is_reliable,
        }


def compute_information_coefficient(
    indicator_values: list[float],
    future_returns: list[float],
    method: str = "pearson",
    min_sample_size: int = MIN_RELIABLE_SAMPLE_SIZE,
    indicator_name: Optional[str] = None,
    scenario: Optional[str] = None,
) -> ICResult:
    """计算某个子指标历史取值跟未来实际收益之间的信息系数（IC）。

    Parameters
    ----------
    indicator_values : 历史上每次信号触发时，该子指标的原始值（或标准化
        子分数，两者都可以传，算出来的只是相关系数，跟单位无关）。
    future_returns : 对应每次信号触发后的未来实际收益率，长度、顺序需要
        跟 `indicator_values` 一一对应（第 i 个指标值对应第 i 次信号的
        未来收益）。
    method : "pearson"（默认，线性相关系数）或 "spearman"（秩相关系数，
        对非线性单调关系更稳健，是量化领域"IC"更常见的定义）。

        ⚠️ 规格书原文只写"计算其历史取值与未来实际收益的信息系数（IC）"，
        没有指明具体用皮尔逊还是斯皮尔曼相关系数——这是我需要你确认的一
        个点：量化因子研究里"IC"更多情况下默认指斯皮尔曼秩相关（对异常值
        更稳健，且不要求线性关系），但也有直接用皮尔逊的用法。默认选了
        更直白的 "pearson"（跟任务描述"输出...相关系数"的字面表述更贴近），
        如果你想要行业更常见的 Spearman 口径，传 `method="spearman"`
        即可，两种都实现了。
    min_sample_size : 判定 `is_reliable` 的样本量门槛，默认 30
        （规格书 6.3 节"低于 30 次"）。
    indicator_name, scenario : 仅用于填充返回结果里的留痕字段，不参与计算。

    Returns
    -------
    ICResult(sample_size, correlation_ic, is_reliable, ...)

    - `sample_size` < 2，或者其中一条序列方差为 0（例如指标值全部相同）
      时，相关系数在数学上没有定义，`correlation_ic` 为 `None`（不是 0——
      0 代表"确实算出了不相关"，`None` 代表"算不出来"，两者含义不同）。
    - `is_reliable` = `sample_size >= min_sample_size`，跟 `correlation_ic`
      是否为 `None` 是两件独立的事：样本量够但恰好不相关（IC≈0）也是
      `is_reliable=True`；样本量不够，即使算出了一个数字，也要标
      `is_reliable=False`（6.3 节"标注为'样本不足，暂不采信'"），调用方
      应该用 `is_reliable` 而不是自己再判断 `sample_size`。
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method 必须是 'pearson' 或 'spearman'，收到：{method!r}")
    if len(indicator_values) != len(future_returns):
        raise ValueError("indicator_values 与 future_returns 长度必须一致")

    sample_size = len(indicator_values)
    is_reliable = sample_size >= min_sample_size

    correlation_ic = None
    if sample_size >= 2:
        x = pd.Series(indicator_values, dtype=float)
        y = pd.Series(future_returns, dtype=float)
        if method == "spearman":
            x, y = x.rank(), y.rank()

        if x.std() > 0 and y.std() > 0:
            raw = float(np.corrcoef(x, y)[0, 1])
            correlation_ic = None if np.isnan(raw) else raw
        # 任一序列方差为 0（所有值相同）：相关系数无定义，保持 None。

    return ICResult(
        sample_size=sample_size,
        correlation_ic=correlation_ic,
        is_reliable=is_reliable,
        indicator_name=indicator_name,
        scenario=scenario,
    )
