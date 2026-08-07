"""第二批 —— 波段趋势指标：ATR、相对强度 RS、简单移动平均。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均（Simple Moving Average）。

    这是 `engine/pipeline` 编排层补的一个基础工具函数：`score_intraday` 的
    均线排列（短/长期均线）、`score_swing` 的中长期均线趋势（50/200 日）、
    `score_leveraged_etf` 的 200 日均线位置，都需要一个"收盘价 N 日滚动
    平均"作为输入，但此前 `engine.indicators` 里没有任何函数计算它——`atr`
    内部虽然也用了 `rolling().mean()`，但那是算真实波幅的滑动平均，不是价格
    本身的移动平均，不能复用。

    之所以现在补在 `engine.indicators.trend` 而不是直接写进编排层
    （`engine/pipeline`），是因为编排层的原则是"纯粹调用组装，不引入新的
    指标计算"——而移动平均恰恰是一个真正的指标计算（哪怕公式极其简单、
    没有任何主观参数选择的空间），按项目现有的分层方式，指标计算都应该在
    `engine.indicators` 这一层，编排层只负责调用它、把结果传给 `engine.scoring`。

    Parameters
    ----------
    series : 输入序列，通常是收盘价（也可以是任何数值序列，例如成交量）。
    period : 滑动窗口天数。

    Returns
    -------
    Series，与输入对齐；前 period-1 个位置因窗口未满为 NaN。
    """
    return series.rolling(window=period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """真实波幅 True Range。

    TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)

    第一根 K 线没有前一日收盘价，用 High - Low 代替。
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()

    tr = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    tr.iloc[0] = range1.iloc[0]  # 首根无前收盘价，退化为 High-Low
    return tr


def atr(df: pd.DataFrame, period: int = 14, method: str = "sma") -> pd.Series:
    """平均真实波幅 ATR，衡量波动性大小（用于止损、仓位管理）。

    Parameters
    ----------
    df : 需包含 high/low/close 列。
    period : 平滑周期，默认 14。
    method : "sma" 简单滑动平均；"wilder" Wilder 平滑
        （ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period，
        首个值用前 period 根 TR 的简单平均初始化）。

    Returns
    -------
    Series，与 df 索引对齐；不足 period 根 K 线的位置为 NaN
    （wilder 方法从第 period 个值才开始有效数值）。
    """
    tr = true_range(df)

    if method == "sma":
        return tr.rolling(window=period, min_periods=period).mean()
    elif method == "wilder":
        result = pd.Series(index=tr.index, dtype=float)
        if len(tr) < period:
            return result
        result.iloc[period - 1] = tr.iloc[:period].mean()
        for i in range(period, len(tr)):
            result.iloc[i] = (result.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
        return result
    raise ValueError("method 必须是 'sma' 或 'wilder'")


def rsi(df: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
    """相对强弱指标 RSI（Relative Strength Index）。

    RSI = 100 × 平均涨幅 / (平均涨幅 + 平均跌幅)，等价于教科书写法
    `100 - 100/(1+RS)`（RS = 平均涨幅/平均跌幅），这里直接用等价的除法
    形式实现，好处是分母 `平均涨幅+平均跌幅` 为 0（窗口内价格完全没变化）
    时不会除以 0 后再除以 0 两次出问题，只需要处理一次分母为 0 的情况。

    这是规格书第 13.3 节记录的已知代码缺口：此前 `engine.indicators` 里
    从未实现 RSI，`score_intraday` 的 RSI 子指标（5.2 节权重 20%）只能靠
    编排层把 `rsi` 当作外部传入的可选参数，缺失时不参与打分。这个函数
    补上后，`engine/pipeline/intraday.py` 会在调用方没有显式传 `rsi` 时
    自动用它从原始 OHLCV 算出来。

    Parameters
    ----------
    df : 需包含 `close` 列。
    period : 平滑周期，默认 14（RSI 最经典、最常用的默认周期）。
    method : "wilder"（默认）或 "sma"。

        - "wilder"：Wilder 本人提出 RSI 时用的原始平滑方法，也是业内最
          常见的默认实现——平均涨幅/跌幅按跟 `atr(method="wilder")` 完全
          相同的递推公式平滑（`avg_t = (avg_{t-1}×(period-1) + x_t) /
          period`），首个值用前 `period` 个涨跌幅的简单平均初始化。
        - "sma"：平均涨幅/跌幅都用简单滑动平均（`rolling().mean()`），
          跟 `atr(method="sma")` 是同一种平滑方式，只是应用在涨跌幅上
          而不是真实波幅上。

        两种方法在第一个有效值上完全一致（都是"前 period 个涨跌幅的简单
        平均"），从第二个有效值开始才会分化——Wilder 平滑对历史值的权重
        衰减更慢（更平滑），简单滑动平均只看窗口内的值（对新数据更敏感）。

    Returns
    -------
    Series，与 `df` 索引对齐，取值范围 [0,100]；窗口不足 `period+1` 根
    K 线（计算涨跌幅需要用到前一根收盘价，所以比 `atr` 多需要 1 根）的
    位置为 NaN。窗口内价格完全没有变化（平均涨幅和平均跌幅都是 0，
    0/0 无定义）时，约定 RSI=50（中性，无涨跌可言，不是"没数据"，所以
    不是 NaN）。
    """
    if method not in ("sma", "wilder"):
        raise ValueError("method 必须是 'sma' 或 'wilder'")

    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    if method == "sma":
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
    else:  # wilder
        avg_gain = pd.Series(index=close.index, dtype=float)
        avg_loss = pd.Series(index=close.index, dtype=float)
        if len(close) >= period + 1:
            avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
            avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()
            for i in range(period + 1, len(close)):
                avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    denom = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 100 * avg_gain / denom
    result[denom == 0] = 50.0
    return result


def relative_strength_ratio(asset: pd.Series, benchmark: pd.Series) -> pd.Series:
    """相对强度比率（规格书 4.6 节"简单版"）= 资产价格 / 基准价格。

    仅为比值本身，未做均线归一化，是 `relative_strength`（Mansfield RS）
    计算中间步骤所依赖的基础序列。
    """
    if (benchmark == 0).any():
        raise ValueError("benchmark 中不能包含 0，无法计算比率")
    return asset / benchmark


def relative_strength(asset: pd.Series, benchmark: pd.Series, period: int = 200) -> pd.Series:
    """Mansfield RS（规格书 4.6 节"更常用"版本）。

    RS 值 = [(资产价/基准价) ÷ (资产价/基准价的 N 日均线) − 1] × 100

    与单纯的比值（`relative_strength_ratio`）不同，Mansfield RS 用比值自身
    的 N 日均线做基准，衡量当前比值相对其近期均值的偏离程度：
    - RS > 0：当前比值高于其 N 日均线，资产正在跑赢基准（RS 线上升）；
    - RS < 0：当前比值低于其 N 日均线，资产正在跑输基准；
    - 与两者各自的绝对价格走势无关，只看比值相对自身均线的偏离。

    Parameters
    ----------
    asset : 资产价格序列。
    benchmark : 基准（大盘或板块指数）价格序列，长度、索引需与 asset 对齐。
    period : 计算比值均线的周期 N，默认 200（可按场景调整，例如日线波段场景
        常用 200 日，如需更短周期可自行传入，如 50）。

    Returns
    -------
    Series，与输入对齐；前 period-1 个位置因均线尚未形成而为 NaN。
    """
    ratio = relative_strength_ratio(asset, benchmark)
    ratio_ma = ratio.rolling(window=period, min_periods=period).mean()
    return (ratio / ratio_ma - 1) * 100
