"""第 6 章 —— 回测框架公共基础设施。

`SignalOutcome` 是四个场景判定函数的统一输出结构，字段名直接对应
`data/schema.sql` 里 `backtest_signals` 表的列，方便以后真实回测直接
`INSERT`：

    outcome          -> backtest_signals.outcome           ('win'/'loss'/'undecided')
    exit_reason      -> backtest_signals.exit_reason        ('target_hit'/'stop_hit'/'time_exit'/'undecided')
    exit_price       -> backtest_signals.exit_price
    exit_ts          -> backtest_signals.exit_ts
    raw_return_pct   -> backtest_signals.raw_return_pct
    excess_return_pct -> backtest_signals.excess_return_pct（仅波段/杠杆ETF场景有值）
    max_drawdown_pct -> backtest_signals.max_drawdown_pct（仅期权/杠杆ETF场景有值）
    extra            -> backtest_signals.metadata（JSONB，场景特有字段，比如夏普比率）

不是每个场景都会填满所有字段——schema 里这些列本来就是可空的，某场景不
适用的字段留 `None`，跟 schema 的设计意图一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SignalOutcome:
    """一次信号的判定结果，字段对齐 `backtest_signals` 表。"""

    outcome: str  # 'win' / 'loss' / 'undecided'
    exit_reason: str  # 'target_hit' / 'stop_hit' / 'time_exit' / 'undecided'
    exit_price: Optional[float] = None
    exit_ts: Optional[Any] = None  # 通常是 price_path 的索引值（时间戳或序号）
    raw_return_pct: Optional[float] = None
    excess_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    extra: dict = field(default_factory=dict)  # 场景特有字段，对应 metadata JSONB 列

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "exit_reason": self.exit_reason,
            "exit_price": self.exit_price,
            "exit_ts": self.exit_ts,
            "raw_return_pct": self.raw_return_pct,
            "excess_return_pct": self.excess_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "metadata": self.extra,
        }


def max_drawdown_from_returns(returns: list[float]) -> float:
    """给定一串收益率序列，算最大回撤（负数或 0，0 代表从未低于起点）。

    做法：把收益率累乘成净值曲线（起点=1），最大回撤 = min((当前净值 -
    历史最高净值) / 历史最高净值)，取值范围 (-1, 0]。这是业界标准定义，
    不是本项目自定义的公式。
    """
    if not returns:
        return 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        max_dd = min(max_dd, drawdown)
    return max_dd


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Optional[float]:
    """年化夏普比率 = (平均超额收益 / 收益率标准差) × √(年化周期数)。

    这是业界标准公式，不是本项目自定义的东西。`risk_free_rate` 是年化
    无风险利率，先换算成跟 `returns` 同周期的无风险收益率再相减。

    收益率序列长度 < 2 或标准差为 0（例如收益率序列全部相同）时返回
    `None`（夏普比率在这种情况下没有统计意义，不是 0）。
    """
    n = len(returns)
    if n < 2:
        return None

    mean_return = sum(returns) / n
    variance = sum((r - mean_return) ** 2 for r in returns) / (n - 1)
    std = variance**0.5
    if std == 0:
        return None

    period_risk_free = risk_free_rate / periods_per_year
    return (mean_return - period_risk_free) / std * (periods_per_year**0.5)


def cumulative_return(returns: list[float]) -> float:
    """把日收益率序列累乘成期末总收益率（例如 [0.01,-0.02] -> 1.01*0.98-1）。"""
    equity = 1.0
    for r in returns:
        equity *= 1.0 + r
    return equity - 1.0
