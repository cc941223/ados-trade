"""sma() 单元测试 —— 手算对照。

这是为 engine/pipeline 编排层补的基础工具函数（详见 sma() 文档里的说明：
score_intraday/score_swing/score_leveraged_etf 都需要收盘价的 N 日移动
平均作为输入，但 engine.indicators 里此前没有这个函数）。
"""
import pandas as pd

from engine.indicators.trend import sma


def test_sma_manual():
    # [1,2,3,4,5,6]，period=3
    # sma[2] = mean(1,2,3) = 2
    # sma[3] = mean(2,3,4) = 3
    # sma[4] = mean(3,4,5) = 4
    # sma[5] = mean(4,5,6) = 5
    series = pd.Series([1, 2, 3, 4, 5, 6])
    result = sma(series, period=3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2
    assert result.iloc[3] == 3
    assert result.iloc[4] == 4
    assert result.iloc[5] == 5


def test_sma_period_one_equals_series_itself():
    series = pd.Series([10.0, 20.0, 30.0])
    result = sma(series, period=1)
    assert list(result) == [10.0, 20.0, 30.0]


def test_sma_period_longer_than_series_is_all_nan():
    series = pd.Series([1.0, 2.0])
    result = sma(series, period=5)
    assert result.isna().all()
