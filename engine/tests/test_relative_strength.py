"""相对强度 RS 单元测试 —— 手算对照。"""
import pandas as pd

from engine.indicators.trend import relative_strength, relative_strength_normalized


def test_relative_strength_manual():
    # asset: 100 -> 110 -> 121（每日 +10%）
    # benchmark: 50 -> 52 -> 54
    # RS = asset/benchmark:
    #   100/50 = 2.0
    #   110/52 = 2.115384615...
    #   121/54 = 2.240740740...
    asset = pd.Series([100, 110, 121])
    benchmark = pd.Series([50, 52, 54])

    rs = relative_strength(asset, benchmark)
    assert abs(rs.iloc[0] - 2.0) < 1e-9
    assert abs(rs.iloc[1] - 2.1153846153846154) < 1e-9
    assert abs(rs.iloc[2] - 2.2407407407407407) < 1e-9


def test_relative_strength_normalized_manual():
    # 以起点归一化到 100：
    #   t0: 2.0/2.0*100 = 100
    #   t1: 2.115384.../2.0*100 = 105.7692...
    #   t2: 2.240740.../2.0*100 = 112.0370...
    asset = pd.Series([100, 110, 121])
    benchmark = pd.Series([50, 52, 54])

    rs_norm = relative_strength_normalized(asset, benchmark)
    assert abs(rs_norm.iloc[0] - 100.0) < 1e-9
    assert abs(rs_norm.iloc[1] - 105.76923076923077) < 1e-6
    assert abs(rs_norm.iloc[2] - 112.03703703703704) < 1e-6


def test_relative_strength_benchmark_zero_raises():
    import pytest

    with pytest.raises(ValueError):
        relative_strength(pd.Series([100, 110]), pd.Series([0, 52]))
