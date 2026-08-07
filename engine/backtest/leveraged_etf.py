"""6.2 节 —— 杠杆 ETF Regime 判断成功评估：evaluate_leveraged_etf_regime()。

评估对象是持续的配置决策而非单笔交易：对比"按 Regime 建议配置比例的组合
收益"与"纯买入持有基准"的收益，评估标准是风险调整后收益（夏普比率）与
最大回撤，而非单看收益率。
"""
from __future__ import annotations

from .base import SignalOutcome, cumulative_return, max_drawdown_from_returns, sharpe_ratio


def evaluate_leveraged_etf_regime(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> SignalOutcome:
    """评估一段时间内"按 Regime 动态配置"策略相对"纯买入持有"基准的表现。

    Parameters
    ----------
    strategy_returns : 按 Regime 建议配置比例调整后的组合日收益率序列
        （例如 Bull regime 满仓杠杆 ETF、Chop regime 减仓、Bear regime
        空仓或反向）。这个序列本身怎么根据历史 Regime 标签算出来，是
        编排层的事，本函数只接收算好的收益率序列。
    benchmark_returns : 同期纯买入持有该杠杆 ETF（不做任何 Regime 调整）
        的日收益率序列，需要跟 `strategy_returns` 长度、顺序完全对应
        （同一批交易日）。
    risk_free_rate, periods_per_year : 传给 `sharpe_ratio` 的年化无风险
        利率与年化周期数，默认周期数 252（美股全年交易日数的常用近似）。

    Returns
    -------
    SignalOutcome：

    - `outcome`："win" 需要同时满足两个条件（AND，缺一不可）：
      1. 策略夏普比率 > 基准夏普比率（风险调整后收益跑赢基准）；
      2. 策略最大回撤 不劣于 基准最大回撤，即 `strategy_max_dd >=
         benchmark_max_dd`（两者都是 ≤0 的负数，"不劣于"是数值上更接近
         0 或更大，代表回撤更轻）。
      任一条件不满足就判 "loss"。

      这是按规格书 6.2 节原文"评估标准为风险调整后收益（如夏普比率）与
      最大回撤，而非单看收益率——杠杆 ETF 波动巨大，Regime 判断的价值
      应体现在回撤更小，而非仅收益更高"实现的：如果只看夏普比率，回撤
      这个指标就成了摆设（算出来但不影响任何判定），跟规格书强调的重点
      矛盾，所以改成两个条件都要满足的 AND 逻辑。
    - 两条收益率序列长度不足 2（夏普比率算不出来）时，`outcome` 为
      "undecided"。
    - `raw_return_pct`：策略累计收益率。`excess_return_pct`：策略累计
      收益率 － 基准累计收益率。`max_drawdown_pct`：策略自身最大回撤。
    - `extra` 额外存了 `sharpe_ratio`/`benchmark_sharpe_ratio`/
      `benchmark_max_drawdown_pct`，对应 `backtest_signals.metadata`
      （JSONB，场景特有字段的存放位置）。
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("strategy_returns 与 benchmark_returns 长度必须一致")

    strategy_sharpe = sharpe_ratio(strategy_returns, risk_free_rate, periods_per_year)
    benchmark_sharpe = sharpe_ratio(benchmark_returns, risk_free_rate, periods_per_year)

    strategy_return = cumulative_return(strategy_returns)
    benchmark_return = cumulative_return(benchmark_returns)
    strategy_max_dd = max_drawdown_from_returns(strategy_returns)
    benchmark_max_dd = max_drawdown_from_returns(benchmark_returns)

    if strategy_sharpe is None or benchmark_sharpe is None:
        outcome = "undecided"
    else:
        sharpe_better = strategy_sharpe > benchmark_sharpe
        drawdown_not_worse = strategy_max_dd >= benchmark_max_dd
        outcome = "win" if (sharpe_better and drawdown_not_worse) else "loss"

    return SignalOutcome(
        outcome=outcome,
        exit_reason="time_exit",
        raw_return_pct=strategy_return,
        excess_return_pct=strategy_return - benchmark_return,
        max_drawdown_pct=strategy_max_dd,
        extra={
            "sharpe_ratio": strategy_sharpe,
            "benchmark_sharpe_ratio": benchmark_sharpe,
            "benchmark_max_drawdown_pct": benchmark_max_dd,
        },
    )
