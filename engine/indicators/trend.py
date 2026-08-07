"""第二批 —— 波段趋势指标：ATR、相对强度 RS。"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
