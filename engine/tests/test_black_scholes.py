"""Black-Scholes 定价与 Greeks 单元测试。

单独校验点：
1. 用教科书经典案例（Hull《期权、期货及其他衍生产品》欧式期权定价示例）核对
   bs_price 与 Delta 计算是否正确。
2. 平值期权（ATM，S=K）在到期时间很短、无风险利率为 0 时，Delta 应非常接近 0.5
   （因为 d1 -> 0，N(0)=0.5）。
3. implied_volatility 反推能还原出用于生成价格的 sigma。
"""
import math

import pytest

from engine.indicators.options import bs_greeks, bs_price, implied_volatility


def test_bs_price_matches_textbook_example():
    # Hull 教材经典例子：S=42, K=40, r=10%, sigma=20%, T=0.5 年
    # 理论 Call 价格 ≈ 4.76，Delta = N(d1) ≈ 0.7791
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20

    call_price = bs_price(S, K, T, r, sigma, "call")
    assert abs(call_price - 4.76) < 0.01

    greeks = bs_greeks(S, K, T, r, sigma, "call")
    assert abs(greeks.delta - 0.7791) < 0.001


def test_atm_call_delta_close_to_half():
    # 平值期权（S=K），无风险利率为 0，到期时间很短时，
    # d1 = (0 + 0.5*sigma^2*T) / (sigma*sqrt(T)) -> 0，Delta = N(d1) -> 0.5
    S = K = 100.0
    T = 0.01
    r = 0.0
    sigma = 0.20

    greeks = bs_greeks(S, K, T, r, sigma, "call")
    assert abs(greeks.delta - 0.5) < 0.02


def test_atm_put_delta_close_to_negative_half():
    S = K = 100.0
    T = 0.01
    r = 0.0
    sigma = 0.20

    greeks = bs_greeks(S, K, T, r, sigma, "put")
    assert abs(greeks.delta - (-0.5)) < 0.02


def test_put_call_delta_relationship():
    # 同参数下 Put Delta = Call Delta - 1
    S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
    call_greeks = bs_greeks(S, K, T, r, sigma, "call")
    put_greeks = bs_greeks(S, K, T, r, sigma, "put")
    assert abs(put_greeks.delta - (call_greeks.delta - 1)) < 1e-9


def test_gamma_and_vega_are_equal_for_call_and_put():
    # 相同参数下，Call 和 Put 的 Gamma、Vega 理论上相等
    S, K, T, r, sigma = 100, 95, 0.75, 0.02, 0.3
    call_greeks = bs_greeks(S, K, T, r, sigma, "call")
    put_greeks = bs_greeks(S, K, T, r, sigma, "put")

    assert abs(call_greeks.gamma - put_greeks.gamma) < 1e-9
    assert abs(call_greeks.vega - put_greeks.vega) < 1e-9
    assert call_greeks.gamma > 0
    assert call_greeks.vega > 0


def test_implied_volatility_recovers_input_sigma():
    S, K, T, r = 100, 100, 0.5, 0.02
    true_sigma = 0.28
    price = bs_price(S, K, T, r, true_sigma, "call")

    recovered_sigma = implied_volatility(price, S, K, T, r, "call")
    assert abs(recovered_sigma - true_sigma) < 1e-6


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.01, 0.2, "invalid")
