"""VWAP 单元测试 —— 3 根 K 线手算对照。"""
import pandas as pd

from engine.indicators.base import vwap


def test_vwap_manual_three_bars():
    # 手算：
    # bar1 典型价 = (10+8+9)/3 = 9,  V=100
    # bar2 典型价 = (11+9+10)/3 = 10, V=200
    # bar3 典型价 = (12+10+11)/3 = 11, V=100
    #
    # VWAP_1 = (9*100) / 100 = 9.0
    # VWAP_2 = (9*100 + 10*200) / 300 = 2900/300 = 9.666666...
    # VWAP_3 = (2900 + 11*100) / 400 = 4000/400 = 10.0
    df = pd.DataFrame(
        {
            "high": [10, 11, 12],
            "low": [8, 9, 10],
            "close": [9, 10, 11],
            "volume": [100, 200, 100],
        }
    )
    result = vwap(df)

    assert result.iloc[0] == 9.0
    assert abs(result.iloc[1] - 9.666666666666666) < 1e-9
    assert result.iloc[2] == 10.0


def test_vwap_empty_dataframe():
    df = pd.DataFrame(columns=["high", "low", "close", "volume"])
    result = vwap(df)
    assert result.empty
