"""IV Term Structure 单元测试 —— 手算对照。"""
from engine.indicators.options import iv_term_structure


def test_iv_term_structure_backwardation_manual():
    # 到期期限（年化）：30天/60天/90天，ATM IV 分别为 0.35 / 0.30 / 0.25
    # 近月 IV 高于远月 -> 逆向期限结构（backwardation）
    # near_far_ratio = 0.35 / 0.25 = 1.4（手算可直接验证）
    expiries_T = [30 / 365, 60 / 365, 90 / 365]
    atm_ivs = [0.35, 0.30, 0.25]

    result = iv_term_structure(expiries_T, atm_ivs)

    assert result["slope"] < 0
    assert abs(result["near_far_ratio"] - 1.4) < 1e-9
    assert result["structure"] == "backwardation"


def test_iv_term_structure_contango_manual():
    # 近月 IV 低于远月 -> 正向期限结构（contango）
    # near_far_ratio = 0.20 / 0.30 = 0.6666...
    expiries_T = [30 / 365, 90 / 365]
    atm_ivs = [0.20, 0.30]

    result = iv_term_structure(expiries_T, atm_ivs)

    assert result["slope"] > 0
    assert abs(result["near_far_ratio"] - (0.20 / 0.30)) < 1e-9
    assert result["structure"] == "contango"


def test_iv_term_structure_requires_two_points():
    import pytest

    with pytest.raises(ValueError):
        iv_term_structure([30 / 365], [0.2])
