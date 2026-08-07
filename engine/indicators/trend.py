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


def relative_strength(asset: pd.Series, benchmark: pd.Series) -> pd.Series:
    """相对强度比率 RS = 资产价格 / 基准价格（例如个股 vs 大盘指数）。

    RS 上升代表资产跑赢基准，下降代表跑输基准，与两者各自涨跌幅大小无关，
    只看比值的相对变化趋势。
    """
    if (benchmark == 0).any():
        raise ValueError("benchmark 中不能包含 0，无法计算比率")
    return asset / benchmark


def relative_strength_normalized(asset: pd.Series, benchmark: pd.Series, base: float = 100.0) -> pd.Series:
    """归一化相对强度线：以序列起点为基准，缩放到 `base`（默认 100）。

    即 RS_norm_t = (asset_t / benchmark_t) / (asset_0 / benchmark_0) * base，
    方便跨品种比较趋势斜率，而不用关心起点比值的绝对大小。
    """
    rs = relative_strength(asset, benchmark)
    if rs.iloc[0] == 0:
        raise ValueError("起点相对强度为 0，无法归一化")
    return rs / rs.iloc[0] * base
