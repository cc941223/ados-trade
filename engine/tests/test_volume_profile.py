"""Volume Profile / POC 单元测试。"""
import pandas as pd

from engine.indicators.base import poc, volume_profile


def test_volume_profile_and_poc_manual():
    # 3 根 K 线，典型价分别为 100 / 100 / 200，
    # 用显式分桶边界 [0, 150, 250]，使结果可手算：
    #   桶 (0,150] 汇聚了两根典型价=100 的 K 线，总量 1000+500=1500
    #   桶 (150,250] 只有典型价=200 的一根，总量 100
    # POC 应落在成交量更大的 (0,150] 桶，桶中点 = (0+150)/2 = 75
    df = pd.DataFrame(
        {
            "high": [100, 100, 200],
            "low": [100, 100, 200],
            "close": [100, 100, 200],
            "volume": [1000, 500, 100],
        }
    )

    profile = volume_profile(df, bins=[0, 150, 250])
    assert len(profile) == 2
    assert profile.iloc[0] == 1500  # (0,150] 桶
    assert profile.iloc[1] == 100  # (150,250] 桶

    poc_price = poc(df, bins=[0, 150, 250])
    assert poc_price == 75.0


def test_poc_empty_dataframe():
    df = pd.DataFrame(columns=["high", "low", "close", "volume"])
    assert volume_profile(df).empty
    import math

    assert math.isnan(poc(df))
