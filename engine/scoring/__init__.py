"""ADOS Engine —— 评分模块（技术规格书第 5 章）。

按 5.2-5.5 节四个场景各暴露一个评分函数，均输出统一的
`ScoreResult(bull_score, bear_score, confidence_score, sub_scores, extra)`：

- `score_intraday`：日内短线
- `score_swing`：波段 / 趋势跟踪
- `score_options`：期权策略
- `score_leveraged_etf`：杠杆 ETF 对冲（Regime 判断）

所有指标原始值均来自 `engine.indicators`，本模块只做标准化 + 加权聚合，
不重新计算任何指标。
"""
from .base import ScoreResult, SubScore, aggregate_scores, linear_map, resolve_weights
from .intraday import score_intraday
from .leveraged_etf import classify_regime, score_leveraged_etf
from .options import score_options
from .swing import score_swing

__all__ = [
    "ScoreResult",
    "SubScore",
    "aggregate_scores",
    "linear_map",
    "resolve_weights",
    "score_intraday",
    "score_swing",
    "score_options",
    "score_leveraged_etf",
    "classify_regime",
]
