"""相对强度 RS 单元测试 —— 手算对照。

按规格书 4.6 节，RS 主用 Mansfield RS 公式：
    RS 值 = [(资产价/基准价) ÷ (资产价/基准价的 N 日均线) − 1] × 100
"""
import pandas as pd

from engine.indicators.trend import relative_strength, relative_strength_ratio


def test_relative_strength_ratio_manual():
    # 简单版比值（规格书 4.6 节"简单版"）：
    # asset: 100 -> 110 -> 121，benchmark: 50 -> 52 -> 54
    # ratio = asset/benchmark = 2.0 / 2.115384... / 2.240740...
    asset = pd.Series([100, 110, 121])
    benchmark = pd.Series([50, 52, 54])

    ratio = relative_strength_ratio(asset, benchmark)
    assert abs(ratio.iloc[0] - 2.0) < 1e-9
    assert abs(ratio.iloc[1] - 2.1153846153846154) < 1e-9
    assert abs(ratio.iloc[2] - 2.2407407407407407) < 1e-9


def test_relative_strength_mansfield_manual():
    # 用 5 个点、period=3 手算 Mansfield RS：
    # asset/benchmark 比值序列：
    #   ratio = [2.0, 2.1, 2.2, 2.4, 2.1]
    # 3 日均线（rolling mean，period=3）：
    #   ma[2] = mean(2.0,2.1,2.2) = 2.1
    #   ma[3] = mean(2.1,2.2,2.4) = 2.233333...
    #   ma[4] = mean(2.2,2.4,2.1) = 2.233333...
    # RS = (ratio/ma - 1) * 100：
    #   RS[2] = (2.2/2.1 - 1) * 100 = 4.761904761904759
    #   RS[3] = (2.4/2.233333... - 1) * 100 = 7.462686567164163
    #   RS[4] = (2.1/2.233333... - 1) * 100 = -5.970149253731331
    asset = pd.Series([100.0, 105.0, 110.0, 120.0, 105.0])
    benchmark = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0])
    # ratio = asset/benchmark = [2.0, 2.1, 2.2, 2.4, 2.1]

    rs = relative_strength(asset, benchmark, period=3)

    assert pd.isna(rs.iloc[0])
    assert pd.isna(rs.iloc[1])
    assert abs(rs.iloc[2] - 4.761904761904759) < 1e-6
    assert abs(rs.iloc[3] - 7.462686567164163) < 1e-6
    assert abs(rs.iloc[4] - (-5.970149253731331)) < 1e-6


def test_relative_strength_positive_means_outperforming():
    # 资产比值持续高于其均线 -> RS 应为正（跑赢基准）
    asset = pd.Series([100.0, 110.0, 121.0, 133.1])
    benchmark = pd.Series([100.0, 100.0, 100.0, 100.0])

    rs = relative_strength(asset, benchmark, period=2)
    valid = rs.dropna()
    assert (valid > 0).all()


def test_relative_strength_ratio_benchmark_zero_raises():
    import pytest

    with pytest.raises(ValueError):
        relative_strength_ratio(pd.Series([100, 110]), pd.Series([0, 52]))
