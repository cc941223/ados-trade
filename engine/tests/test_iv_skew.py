"""IV Skew（25-delta 风险逆转）单元测试。

按规格书 4.6 节：skew = IV(25-delta 虚值 Put) − IV(25-delta 虚值 Call)，
25-delta 位置需要用 Black-Scholes 反推 Delta 后再定位，不是直接取最接近
现价的行使价。

下面的期权链 Call Delta（用 `bs_greeks` 算出，该函数已在
test_black_scholes.py 中用教科书案例核对过）：

    S=100, T=30/365, r=0.02
    strike=90,  iv=0.32 -> call_delta = 0.8872903228857234
    strike=95,  iv=0.28 -> call_delta = 0.7579102756859026
    strike=100, iv=0.25 -> call_delta = 0.5234330145822786
    strike=105, iv=0.23 -> call_delta = 0.24760992605309645
    strike=110, iv=0.21 -> call_delta = 0.06354426222059294

25-delta Call（Delta=0.25）落在 strike 100（delta=0.5234...）与
strike 105（delta=0.2476...）之间，线性插值：
    call_25d_iv = 0.23 + (0.25-0.24760992...)/(0.52343301...-0.24760992...)*(0.25-0.23)
                = 0.23017330484983317

25-delta Put（等价于 Call Delta=0.75）落在 strike 95（delta=0.7579...）与
strike 100（delta=0.5234...）之间：
    put_25d_iv = 0.25 + (0.75-0.52343301...)/(0.75791028...-0.52343301...)*(0.28-0.25)
               = 0.278987926294174

skew = put_25d_iv - call_25d_iv = 0.048814621444340844
"""
from engine.indicators.options import iv_skew


def test_iv_skew_25_delta_risk_reversal_manual():
    strikes = [90, 95, 100, 105, 110]
    ivs = [0.32, 0.28, 0.25, 0.23, 0.21]
    spot = 100
    T = 30 / 365
    r = 0.02

    result = iv_skew(strikes, ivs, spot, T, r)

    assert abs(result["call_25d_iv"] - 0.23017330484983317) < 1e-6
    assert abs(result["put_25d_iv"] - 0.278987926294174) < 1e-6
    assert abs(result["skew"] - 0.048814621444340844) < 1e-6
    assert result["skew"] > 0  # 左偏：下行保护更贵


def test_iv_skew_flat_smile_gives_zero_skew():
    # 若各行使价 IV 完全相同（无微笑/偏斜），25-delta Call 和 Put 的 IV
    # 应相等，skew ≈ 0。
    strikes = [90, 95, 100, 105, 110]
    ivs = [0.25, 0.25, 0.25, 0.25, 0.25]
    spot = 100
    T = 30 / 365
    r = 0.02

    result = iv_skew(strikes, ivs, spot, T, r)
    assert abs(result["skew"]) < 1e-9
    assert abs(result["call_25d_iv"] - 0.25) < 1e-9
    assert abs(result["put_25d_iv"] - 0.25) < 1e-9


def test_iv_skew_requires_at_least_two_strikes():
    import pytest

    with pytest.raises(ValueError):
        iv_skew([100], [0.25], 100, 30 / 365)
