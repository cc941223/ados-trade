"""IV Skew 单元测试 —— 手算对照。"""
from engine.indicators.options import iv_skew


def test_iv_skew_manual():
    # strikes:  90    95    100   105   110
    # ivs:      0.30  0.28  0.25  0.24  0.23
    # spot = 100
    #
    # ATM 行使价 = 100（与 spot 完全相等），ATM IV = 0.25
    # 低于 spot 且最接近 spot 的行使价 = 95（OTM Put 方向），IV = 0.28
    # skew = 0.28 - 0.25 = 0.03（正值，左偏，下行保护更贵）
    strikes = [90, 95, 100, 105, 110]
    ivs = [0.30, 0.28, 0.25, 0.24, 0.23]
    spot = 100

    result = iv_skew(strikes, ivs, spot)

    assert result["atm_strike"] == 100
    assert result["atm_iv"] == 0.25
    assert result["otm_put_strike"] == 95
    assert result["otm_put_iv"] == 0.28
    assert abs(result["skew"] - 0.03) < 1e-9


def test_iv_skew_no_strike_below_spot():
    # 所有行使价都高于 spot 时，otm_put 退化为 atm
    strikes = [105, 110, 115]
    ivs = [0.20, 0.22, 0.24]
    spot = 100

    result = iv_skew(strikes, ivs, spot)
    assert result["otm_put_strike"] == result["atm_strike"]
    assert result["skew"] == 0.0
