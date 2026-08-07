"""第 6 章 —— 回测框架设计。

四个场景成功判定函数 + IC（信息系数）计算，字段全部对齐
`data/schema.sql` 里 `backtest_signals`/`indicator_correlation_history`
两张表，方便以后真实回测直接写库。
"""
from .base import SignalOutcome, cumulative_return, max_drawdown_from_returns, sharpe_ratio
from .correlation import MIN_RELIABLE_SAMPLE_SIZE, ICResult, compute_information_coefficient
from .intraday import evaluate_intraday_signal
from .leveraged_etf import evaluate_leveraged_etf_regime
from .options import evaluate_options_signal
from .swing import evaluate_swing_signal

__all__ = [
    "SignalOutcome",
    "cumulative_return",
    "max_drawdown_from_returns",
    "sharpe_ratio",
    "ICResult",
    "MIN_RELIABLE_SAMPLE_SIZE",
    "compute_information_coefficient",
    "evaluate_intraday_signal",
    "evaluate_swing_signal",
    "evaluate_options_signal",
    "evaluate_leveraged_etf_regime",
]
