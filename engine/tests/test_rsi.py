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


def test_rsi_wilder_matches_independent_closed_form_deep_into_recursion():
    """前面几个手算测试只有 8 根 K 线（period=5），最多验证了种子值
    （index=5）之后 2 步递推（index=6、7）——如果 Wilder 递推循环本身
    有 bug（比如引用错了上一步的值、denominator 用错、循环边界差一位），
    只要不影响前 2 步就测不出来。这里用一条完全不同的计算路径——把
    Wilder 递推 `avg_t = avg_{t-1}*(period-1)/period + x_t/period` 展开
    成等比权重求和的封闭形式：

        avg_i = seed * w^(i-period) + Σ_{k=period+1}^{i} x_k/period * w^(i-k)
        （w = (period-1)/period）

    而不是逐步循环，在一条 30 根 K 线、period=5（种子之后有 25 步真正
    递归）的序列上独立算出 avg_gain/avg_loss，再跟 rsi() 的输出逐点核对，
    专门验证进入真正递归阶段之后数值没有漂移。
    """
    period = 5
    deltas = [
        2, -3, 5, 2, -3, 5, 3, -1, 4, -2, 6, -4, 1, 3, -5,
        2, 2, -1, 4, -3, 5, -2, 3, 1, -4, 6, -3, 2, -1,
    ]
    closes = [100.0]
    for d in deltas:
        closes.append(closes[-1] + d)
    df = _make_df(closes)

    result = rsi(df, period=period, method="wilder")

    close = pd.Series(closes)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    def _weighted_sum_reference(x):
        w = (period - 1) / period
        seed = x.iloc[1 : period + 1].mean()
        avg = [float("nan")] * len(x)
        avg[period] = seed
        for i in range(period + 1, len(x)):
            total = seed * (w ** (i - period))
            for k in range(period + 1, i + 1):
                total += x.iloc[k] / period * (w ** (i - k))
            avg[i] = total
        return avg

    ref_gain = _weighted_sum_reference(gain)
    ref_loss = _weighted_sum_reference(loss)

    for i in range(period, len(closes)):
        denom = ref_gain[i] + ref_loss[i]
        expected = 50.0 if denom == 0 else 100 * ref_gain[i] / denom
        assert abs(result.iloc[i] - expected) < 1e-9, f"index {i} 不一致"
