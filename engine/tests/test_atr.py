"""ATR 单元测试 —— 4 根 K 线手算对照。"""
import pandas as pd

from engine.indicators.trend import atr, true_range


def _sample_df():
    return pd.DataFrame(
        {
            "high": [10, 11, 12, 13],
            "low": [8, 9, 9, 11],
            "close": [9, 10, 11, 12],
        }
    )


def test_true_range_manual():
    # TR1 = H-L = 10-8 = 2（首根无前收盘价）
    # TR2 = max(11-9=2, |11-9|=2, |9-9|=0) = 2
    # TR3 = max(12-9=3, |12-10|=2, |9-10|=1) = 3
    # TR4 = max(13-11=2, |13-11|=2, |11-11|=0) = 2
    df = _sample_df()
    tr = true_range(df)
    assert list(tr) == [2, 2, 3, 2]


def test_atr_sma_manual():
    # period=3 简单滑动平均：
    # ATR[2] = mean(TR1,TR2,TR3) = (2+2+3)/3 = 2.333...
    # ATR[3] = mean(TR2,TR3,TR4) = (2+3+2)/3 = 2.333...
    df = _sample_df()
    result = atr(df, period=3, method="sma")

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert abs(result.iloc[2] - 7 / 3) < 1e-9
    assert abs(result.iloc[3] - 7 / 3) < 1e-9


def test_atr_wilder_manual():
    # Wilder 平滑：首值 = 前 period 根 TR 简单平均，之后递推
    # ATR[2] = mean(TR1,TR2,TR3) = 2.333...
    # ATR[3] = (ATR[2]*(3-1) + TR4) / 3 = (2.333...*2 + 2)/3 = 2.222...
    df = _sample_df()
    result = atr(df, period=3, method="wilder")

    assert abs(result.iloc[2] - 7 / 3) < 1e-9
    expected_atr3 = (7 / 3 * 2 + 2) / 3
    assert abs(result.iloc[3] - expected_atr3) < 1e-9
