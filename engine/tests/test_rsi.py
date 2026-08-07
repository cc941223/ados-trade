"""rsi() 单元测试 —— 手算对照。

统一用收盘价 [100, 102, 99, 104, 106, 103, 108, 111]，period=5：
    delta = [nan, 2, -3, 5, 2, -3, 5, 3]
    gain  = [nan, 2, 0, 5, 2, 0, 5, 3]
    loss  = [nan, 0, 3, 0, 0, 3, 0, 0]

Wilder 和 SMA 两种方法在第一个有效值（index=5）上完全一致（都是"前 5 个
涨跌幅的简单平均"），从 index=6 开始才会分化，用来验证两种方法确实按
预期各自平滑。
"""
import math

import pandas as pd
import pytest

from engine.indicators.trend import rsi


def _make_df(closes):
    return pd.DataFrame({"close": closes})


CLOSES = [100, 102, 99, 104, 106, 103, 108, 111]


def test_rsi_wilder_manual():
    # index5（第一个有效值，两种方法通用）：
    #   avg_gain = mean(2,0,5,2,0) = 1.8, avg_loss = mean(0,3,0,0,3) = 1.2
    #   RSI = 100*1.8/(1.8+1.2) = 60.0
    # index6（Wilder 递推）：
    #   avg_gain = (1.8*4 + 5)/5 = 2.44, avg_loss = (1.2*4 + 0)/5 = 0.96
    #   RSI = 100*2.44/(2.44+0.96) = 71.76470588235294
    # index7：
    #   avg_gain = (2.44*4 + 3)/5 = 2.552, avg_loss = (0.96*4 + 0)/5 = 0.768
    #   RSI = 100*2.552/(2.552+0.768) = 76.86746987951807
    df = _make_df(CLOSES)
    result = rsi(df, period=5, method="wilder")

    assert all(math.isnan(v) for v in result.iloc[:5])
    assert abs(result.iloc[5] - 60.0) < 1e-9
    assert abs(result.iloc[6] - 71.76470588235294) < 1e-9
    assert abs(result.iloc[7] - 76.86746987951807) < 1e-9


def test_rsi_sma_manual():
    # index5 跟 wilder 完全一样：60.0
    # index6（简单滑动平均，用 gain[2:7]/loss[2:7]）：
    #   avg_gain = mean(0,5,2,0,5) = 2.4, avg_loss = mean(3,0,0,3,0) = 1.2
    #   RSI = 100*2.4/(2.4+1.2) = 66.66666666666667
    # index7（用 gain[3:8]/loss[3:8]）：
    #   avg_gain = mean(5,2,0,5,3) = 3.0, avg_loss = mean(0,0,3,0,0) = 0.6
    #   RSI = 100*3.0/(3.0+0.6) = 83.33333333333333
    df = _make_df(CLOSES)
    result = rsi(df, period=5, method="sma")

    assert abs(result.iloc[5] - 60.0) < 1e-9
    assert abs(result.iloc[6] - 66.66666666666667) < 1e-9
    assert abs(result.iloc[7] - 83.33333333333333) < 1e-9


def test_rsi_default_method_is_wilder():
    df = _make_df(CLOSES)
    default_result = rsi(df, period=5)
    wilder_result = rsi(df, period=5, method="wilder")
    sma_result = rsi(df, period=5, method="sma")

    assert abs(default_result.iloc[6] - wilder_result.iloc[6]) < 1e-9
    # 确认默认值真的是 wilder 而不是碰巧跟 sma 一样（index6 两种方法结果不同）
    assert abs(default_result.iloc[6] - sma_result.iloc[6]) > 1.0


def test_rsi_only_gains_saturates_to_100():
    # 连续上涨，avg_loss 恒为 0，RSI 应该饱和到 100（不是除以0报错）
    df = _make_df([100, 101, 102, 103, 104, 105, 106])
    result = rsi(df, period=5)
    assert result.iloc[5] == 100.0
    assert result.iloc[6] == 100.0


def test_rsi_only_losses_saturates_to_0():
    df = _make_df([106, 105, 104, 103, 102, 101, 100])
    result = rsi(df, period=5)
    assert result.iloc[5] == 0.0


def test_rsi_flat_prices_is_neutral_50_not_nan():
    # 价格完全没变化，avg_gain=avg_loss=0，0/0 无定义，约定为 50（中性），
    # 不是 NaN——这跟"数据不足"的 NaN 含义不同。
    df = _make_df([100.0] * 10)
    result = rsi(df, period=5)
    assert all(math.isnan(v) for v in result.iloc[:5])
    assert all(v == 50.0 for v in result.iloc[5:])


def test_rsi_insufficient_history_is_all_nan():
    # 计算涨跌幅需要用到前一根收盘价，所以至少需要 period+1 根 K 线才能
    # 有第一个有效值；只给 period 根（甚至更少）应该全是 NaN。
    df = _make_df([100, 102, 99, 104, 106])  # 5 根，period=5 需要 6 根才够
    result = rsi(df, period=5)
    assert all(math.isnan(v) for v in result)


def test_rsi_bounded_between_0_and_100():
    df = _make_df(CLOSES)
    for method in ("wilder", "sma"):
        result = rsi(df, period=5, method=method).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()


def test_rsi_invalid_method_raises():
    df = _make_df(CLOSES)
    with pytest.raises(ValueError):
        rsi(df, period=5, method="ema")
