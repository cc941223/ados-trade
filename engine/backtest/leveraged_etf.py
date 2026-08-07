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

    - `outcome`："win" 如果策略夏普比率高于基准夏普比率，否则 "loss"。

      ⚠️ 规格书原文"评估标准为风险调整后收益（如夏普比率）与最大回撤，
      而非单看收益率"只说了要看这两个指标，没有给出"两个指标都要满足
      才算赢"还是"看哪一个"的组合规则。这里的处理是：`outcome` 只按
      夏普比率高低判定（风险调整后收益是否跑赢基准），最大回撤单独存进
      `max_drawdown_pct`（策略自身）和 `extra["benchmark_max_drawdown_pct"]`
      （基准）留痕，供人工核对"是不是回撤更小"，不参与 `outcome` 的判定
      逻辑——这是我的设计选择，规格书没有明确到这个细节，需要你确认，
      如果你认为应该要求"夏普更高且回撤更小才算赢"，告诉我可以改成
      两个条件同时满足的 AND 逻辑。
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
        outcome = "win" if strategy_sharpe > benchmark_sharpe else "loss"

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
