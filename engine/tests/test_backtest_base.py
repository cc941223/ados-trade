"""engine.backtest.base 单元测试 —— max_drawdown_from_returns / sharpe_ratio /
cumulative_return，手算对照。
"""
import math

from engine.backtest.base import cumulative_return, max_drawdown_from_returns, sharpe_ratio


def test_max_drawdown_from_returns_manual():
    # 收益率 [0.10, -0.05, -0.05, 0.02]
    # 净值: 1.0 ->1.10 ->1.045 ->0.99275 ->1.012605
    # 峰值: 1.10 (第1步后) 一直保持到最后
    # 回撤: 0, -0.05, -0.0975, -0.0794...
    # 最大回撤 = -0.0975（第3步）
    returns = [0.10, -0.05, -0.05, 0.02]
    result = max_drawdown_from_returns(returns)
    assert abs(result - (-0.0975)) < 1e-9


def test_max_drawdown_never_negative_returns_zero():
    returns = [0.01, 0.02, 0.03]
    assert max_drawdown_from_returns(returns) == 0.0


def test_max_drawdown_empty_returns_zero():
    assert max_drawdown_from_returns([]) == 0.0


def test_sharpe_ratio_manual():
    # 收益率 [0.01, 0.02, -0.01, 0.015]，risk_free=0
    # mean = 0.00875, std(样本标准差,n-1) = 0.013149778...
    # sharpe = mean/std * sqrt(252) = 10.563063...
    returns = [0.01, 0.02, -0.01, 0.015]
    result = sharpe_ratio(returns)
    assert abs(result - 10.563063630074943) < 1e-6


def test_sharpe_ratio_with_risk_free_rate():
    # risk_free_rate=0.05（年化），periods_per_year=252
    # 每期无风险收益 = 0.05/252 = 0.000198412...
    # mean_excess = 0.00875 - 0.000198412... = 0.008551587...
    returns = [0.01, 0.02, -0.01, 0.015]
    result = sharpe_ratio(returns, risk_free_rate=0.05, periods_per_year=252)
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = variance**0.5
    expected = (mean - 0.05 / 252) / std * (252**0.5)
    assert abs(result - expected) < 1e-9


def test_sharpe_ratio_insufficient_samples_returns_none():
    assert sharpe_ratio([0.01]) is None
    assert sharpe_ratio([]) is None


def test_sharpe_ratio_zero_std_returns_none():
    # 收益率全部相同，标准差为 0，夏普比率没有统计意义
    assert sharpe_ratio([0.01, 0.01, 0.01]) is None


def test_cumulative_return_manual():
    # (1.10)*(0.95)*(0.95)*(1.02) - 1
    returns = [0.10, -0.05, -0.05, 0.02]
    result = cumulative_return(returns)
    expected = 1.10 * 0.95 * 0.95 * 1.02 - 1
    assert abs(result - expected) < 1e-9


def test_cumulative_return_empty_is_zero():
    assert cumulative_return([]) == 0.0
