"""波动损耗率回归 单元测试 —— 构造无噪声数据，手算对照。"""
import numpy as np

from engine.indicators.leveraged import volatility_decay_regression


def test_decay_regression_recovers_exact_linear_relationship():
    # 构造 underlying 日收益率，令 etf_returns = 2 * underlying_returns - 0.001
    # （模拟 2x 杠杆 ETF，每日损耗 0.001，即 -0.1%），无噪声。
    # 最小二乘回归应该精确还原 beta=2.0, alpha(daily_decay)=-0.001, R^2=1.0
    underlying_returns = [0.01, -0.02, 0.03, -0.01, 0.005, -0.015]
    etf_returns = [2 * r - 0.001 for r in underlying_returns]

    result = volatility_decay_regression(etf_returns, underlying_returns, leverage=2.0)

    assert abs(result.beta - 2.0) < 1e-9
    assert abs(result.daily_decay - (-0.001)) < 1e-9
    assert abs(result.r_squared - 1.0) < 1e-9


def test_decay_regression_negative_leverage_inverse_etf():
    # -1x 反向 ETF：etf_returns = -1 * underlying_returns - 0.0005
    underlying_returns = [0.02, -0.01, 0.015, -0.03]
    etf_returns = [-1 * r - 0.0005 for r in underlying_returns]

    result = volatility_decay_regression(etf_returns, underlying_returns, leverage=-1.0)

    assert abs(result.beta - (-1.0)) < 1e-9
    assert abs(result.daily_decay - (-0.0005)) < 1e-9


def test_decay_regression_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        volatility_decay_regression([0.01, 0.02], [0.01], leverage=2.0)


def test_decay_regression_requires_two_points():
    import pytest

    with pytest.raises(ValueError):
        volatility_decay_regression([0.01], [0.01], leverage=2.0)
