"""evaluate_leveraged_etf_regime() 单元测试 —— 手算对照。"""
import pytest

from engine.backtest.leveraged_etf import evaluate_leveraged_etf_regime


def test_strategy_beats_benchmark_sharpe_is_win():
    # strategy 收益率 [0.03,-0.01,0.02,0.01]，benchmark [0.02,-0.02,0.015,0.005]
    # sharpe: strategy≈11.619, benchmark≈4.460 -> strategy更高 -> win
    # 累计收益: strategy≈0.050495, benchmark≈0.019667
    # 最大回撤: strategy≈-0.01, benchmark≈-0.02
    strategy_returns = [0.03, -0.01, 0.02, 0.01]
    benchmark_returns = [0.02, -0.02, 0.015, 0.005]

    result = evaluate_leveraged_etf_regime(strategy_returns, benchmark_returns)

    assert result.outcome == "win"
    assert abs(result.raw_return_pct - 0.0504949400000001) < 1e-9
    assert abs(result.excess_return_pct - (0.0504949400000001 - 0.01966696999999984)) < 1e-9
    assert abs(result.max_drawdown_pct - (-0.009999999999999976)) < 1e-9
    assert abs(result.extra["sharpe_ratio"] - 11.618950038622248) < 1e-6
    assert abs(result.extra["benchmark_sharpe_ratio"] - 4.460351650050169) < 1e-6
    assert abs(result.extra["benchmark_max_drawdown_pct"] - (-0.019999999999999973)) < 1e-9


def test_strategy_worse_sharpe_than_benchmark_is_loss():
    # 交换上面两组数据，strategy 变成夏普更低的那组 -> loss
    strategy_returns = [0.02, -0.02, 0.015, 0.005]
    benchmark_returns = [0.03, -0.01, 0.02, 0.01]

    result = evaluate_leveraged_etf_regime(strategy_returns, benchmark_returns)
    assert result.outcome == "loss"


def test_higher_return_but_lower_sharpe_can_still_lose():
    # 验证判定标准是夏普比率而不是单看收益率：构造一组收益率更高但波动
    # 也更大（夏普更低）的策略，对比一组收益率较低但波动很小（夏普更高）
    # 的基准。
    volatile_high_return = [0.20, -0.15, 0.18, -0.10]
    steady_low_return = [0.01, 0.012, 0.009, 0.011]

    result = evaluate_leveraged_etf_regime(volatile_high_return, steady_low_return)

    # 累计收益应该是 volatile 那组更高
    assert result.raw_return_pct > 0
    assert result.excess_return_pct > 0
    # 但夏普比率应该判定为 loss（波动太大，风险调整后跑输稳定的基准）
    assert result.outcome == "loss"
    assert result.extra["sharpe_ratio"] < result.extra["benchmark_sharpe_ratio"]


def test_insufficient_samples_is_undecided():
    # 只有 1 个观测点，夏普比率算不出来 -> undecided
    result = evaluate_leveraged_etf_regime([0.01], [0.01])
    assert result.outcome == "undecided"
    assert result.extra["sharpe_ratio"] is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_leveraged_etf_regime([0.01, 0.02], [0.01])
