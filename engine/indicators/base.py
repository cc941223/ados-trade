"""第一批 —— 通用基础指标：VWAP、Volume Profile / POC。

输入统一约定：OHLCV 数据用 `pandas.DataFrame`，至少包含列
``high``, ``low``, ``close``, ``volume``（时间顺序需已排好，索引即为
K 线序号或时间戳）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def typical_price(df: pd.DataFrame) -> pd.Series:
    """典型价 = (最高价 + 最低价 + 收盘价) / 3。"""
    return (df["high"] + df["low"] + df["close"]) / 3.0


def vwap(df: pd.DataFrame) -> pd.Series:
    """成交量加权平均价（VWAP），逐 K 线累计计算。

    VWAP_t = Σ(typical_price_i * volume_i) / Σ(volume_i)，i 从序列起点累计到 t。

    Parameters
    ----------
    df : DataFrame，需包含 high/low/close/volume 列。

    Returns
    -------
    Series，与 df 索引对齐，为每一根 K 线截止到当前的累计 VWAP。
    """
    if df.empty:
        return pd.Series(dtype=float)

    tp = typical_price(df)
    cum_pv = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_pv / cum_vol


def volume_profile(df: pd.DataFrame, bins: int = 10, price_col: str | None = None) -> pd.Series:
    """成交量分布（Volume Profile）。

    把价格区间划分成 `bins` 个等宽价格桶，将每根 K 线的成交量计入其
    典型价（或指定 `price_col`）所落入的价格桶，得到每个价格桶的总成交量。

    Parameters
    ----------
    df : DataFrame，需包含 high/low/close/volume 列（若指定 price_col 则用该列代替典型价）。
    bins : 价格桶数量。
    price_col : 若提供，使用该列作为分桶依据的价格，否则使用典型价。

    Returns
    -------
    Series，index 为 pandas.Interval（价格区间），value 为该区间总成交量，
    按区间从低到高排序。
    """
    if df.empty:
        return pd.Series(dtype=float)

    price = df[price_col] if price_col else typical_price(df)
    price_bins = pd.cut(price, bins=bins)
    profile = df.groupby(price_bins)["volume"].sum()
    return profile.sort_index()


def poc(df: pd.DataFrame, bins: int = 10, price_col: str | None = None) -> float:
    """POC（Point of Control）—— 成交量最集中的价格水平。

    取 `volume_profile` 中成交量最大的价格桶，返回该桶的中点价格。
    """
    profile = volume_profile(df, bins=bins, price_col=price_col)
    if profile.empty or profile.sum() == 0:
        return float("nan")
    max_bucket = profile.idxmax()
    return float(max_bucket.mid)
