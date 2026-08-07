"""value_area() 单元测试 —— 手算对照。

统一用显式桶边界 `bins=[0,10,20,30,40,50,...]`，每个价格桶只放一根
"典型价=该桶中点、成交量=目标值"的 K 线，方便精确控制 volume_profile 的
分桶结果，手算验证 value_area 的扩展逻辑。
"""
import math

import pandas as pd

from engine.indicators.base import value_area


def _bars(prices, volumes):
    return pd.DataFrame({"high": prices, "low": prices, "close": prices, "volume": volumes})


def test_value_area_asymmetric_expansion_manual():
    # 5 个桶 (0,10]/(10,20]/(20,30]/(30,40]/(40,50]，成交量 10/30/50/15/5，总量110
    # POC = (20,30]（成交量50）
    # target = 110*0.7 = 77
    # 第1步：左邻桶(10,20]=30 > 右邻桶(30,40]=15，扩展左侧
    #        cumulative = 50+30 = 80 >= 77，停止
    # 覆盖区间 (10,20] ∪ (20,30] -> value_area_low=10, value_area_high=30
    df = _bars([5, 15, 25, 35, 45], [10, 30, 50, 15, 5])

    high, low = value_area(df, bins=[0, 10, 20, 30, 40, 50])

    assert high == 30.0
    assert low == 10.0


def test_value_area_tie_expands_both_sides_manual():
    # 5 个桶，成交量 5/20/50/20/5，总量100，POC=(20,30]
    # target = 100*0.7 = 70
    # 第1步：左邻=20，右邻=20，相同 -> 两侧同时扩展
    #        cumulative = 50+20+20 = 90 >= 70，停止
    # 覆盖区间 (10,20] ∪ (20,30] ∪ (30,40] -> value_area_low=10, value_area_high=40
    df = _bars([5, 15, 25, 35, 45], [5, 20, 50, 20, 5])

    high, low = value_area(df, bins=[0, 10, 20, 30, 40, 50])

    assert high == 40.0
    assert low == 10.0


def test_value_area_multi_step_expansion_manual():
    # 7 个桶 (0,10]...(60,70]，成交量 2/8/15/50/12/9/4，总量100
    # POC = (30,40]（成交量50，第4个桶，下标3）
    # target = 100*0.7 = 70
    # 第1步：左邻(20,30]=15 > 右邻(40,50]=12，扩展左侧
    #        cumulative = 50+15 = 65 < 70，继续
    # 第2步：左邻变为(10,20]=8，右邻仍是(40,50]=12，12>8，扩展右侧
    #        cumulative = 65+12 = 77 >= 70，停止
    # 覆盖区间 (20,30] ∪ (30,40] ∪ (40,50] -> value_area_low=20, value_area_high=50
    df = _bars([5, 15, 25, 35, 45, 55, 65], [2, 8, 15, 50, 12, 9, 4])

    high, low = value_area(df, bins=[0, 10, 20, 30, 40, 50, 60, 70])

    assert high == 50.0
    assert low == 20.0


def test_value_area_single_bucket_returns_its_own_edges():
    df = _bars([5, 5, 5], [10, 20, 30])
    high, low = value_area(df, bins=[0, 10])
    assert high == 10.0
    assert low == 0.0


def test_value_area_percentage_100_covers_all_buckets():
    # percentage=1.0 时 target=总成交量，必然要扩展到覆盖全部价格桶才能达标
    # （target<=总成交量恒成立，所以扩展到底时 cumulative 一定 >= target，
    # 不存在"两侧都扩展完了但还没达标"的情况——这也是本函数的一个不变量）。
    # 5 个桶，成交量 10/5/50/5/30，总量100，target=100
    # 第1步：左邻=5，右邻=5，相同 -> 两侧同时扩展，cumulative=50+5+5=60<100
    #        此时 low=1(桶(10,20]), high=3(桶(30,40])
    # 第2步：左邻=10(桶(0,10])，右邻=30(桶(40,50])，右邻更大 -> 扩展右侧
    #        cumulative=60+30=90<100，此时 high=4 已到达最右边界
    # 第3步：右侧已无邻桶（right_vol=None），只能扩展左侧
    #        cumulative=90+10=100>=100，停止，low=0（已到达最左边界）
    # 覆盖全部 5 个桶 -> value_area_low=0, value_area_high=50
    df = _bars([5, 15, 25, 35, 45], [10, 5, 50, 5, 30])
    high, low = value_area(df, bins=[0, 10, 20, 30, 40, 50], percentage=1.0)
    assert high == 50.0
    assert low == 0.0


def test_value_area_invalid_percentage_raises():
    import pytest

    df = _bars([5, 15], [10, 20])
    with pytest.raises(ValueError):
        value_area(df, bins=[0, 10, 20], percentage=0)
    with pytest.raises(ValueError):
        value_area(df, bins=[0, 10, 20], percentage=1.5)


def test_value_area_empty_dataframe_returns_nan():
    df = pd.DataFrame(columns=["high", "low", "close", "volume"])
    high, low = value_area(df)
    assert math.isnan(high)
    assert math.isnan(low)


def test_value_area_default_percentage_is_70_percent():
    # 默认 percentage=0.7，与显式传入 0.7 的结果一致
    df = _bars([5, 15, 25, 35, 45], [10, 30, 50, 15, 5])
    default_result = value_area(df, bins=[0, 10, 20, 30, 40, 50])
    explicit_result = value_area(df, bins=[0, 10, 20, 30, 40, 50], percentage=0.7)
    assert default_result == explicit_result
