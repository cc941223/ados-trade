"""第一批 —— 通用基础指标：VWAP、Volume Profile / POC / Value Area。

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


def value_area(
    df: pd.DataFrame,
    bins: int = 10,
    price_col: str | None = None,
    percentage: float = 0.7,
) -> tuple[float, float]:
    """Value Area（价值区域）—— 成交量集中的核心价格区间。

    基于 `volume_profile` 的输出，从 POC（成交量最大的价格桶）开始向价格
    两侧交替扩展相邻价格桶：

    1. 每一步比较"向下扩展一个桶"和"向上扩展一个桶"各能新增多少成交量，
       选择增量更大的一侧扩展；增量相同则两侧同时扩展一个桶。
    2. 累计成交量（POC 桶 + 已扩展的桶）达到总成交量的 `percentage`
       （默认 70%）时停止。
    3. 已经扩展到价格区间的边界（某一侧没有更多桶可扩展）时，只能继续
       扩展另一侧，直到达到目标比例或两侧都扩展完。

    Parameters
    ----------
    df : DataFrame，需包含 high/low/close/volume 列（若指定 price_col 则用该列代替典型价）。
    bins : 传给 `volume_profile` 的价格桶数量（或显式桶边界列表）。
    price_col : 若提供，使用该列作为分桶依据的价格，否则使用典型价。
    percentage : Value Area 覆盖的成交量占比，默认 0.7（即 70%）。

    Returns
    -------
    (value_area_high, value_area_low)：Value Area 覆盖的最高价格桶的上边界、
    最低价格桶的下边界。数据为空或 `percentage` 不在 (0,1] 范围内时返回
    `(nan, nan)`。
    """
    if not 0 < percentage <= 1:
        raise ValueError("percentage 必须在 (0, 1] 区间内")

    profile = volume_profile(df, bins=bins, price_col=price_col)
    total_volume = profile.sum()
    if profile.empty or total_volume == 0:
        return float("nan"), float("nan")

    buckets = list(profile.index)
    volumes = list(profile.values)
    n = len(buckets)

    poc_pos = int(np.argmax(volumes))
    low = high = poc_pos
    cumulative = float(volumes[poc_pos])
    target = total_volume * percentage

    while cumulative < target and (low > 0 or high < n - 1):
        left_vol = volumes[low - 1] if low > 0 else None
        right_vol = volumes[high + 1] if high < n - 1 else None

        if left_vol is not None and (right_vol is None or left_vol > right_vol):
            low -= 1
            cumulative += left_vol
        elif right_vol is not None and (left_vol is None or right_vol > left_vol):
            high += 1
            cumulative += right_vol
        else:
            # 两侧增量相同（且都存在）：同时扩展两侧
            low -= 1
            high += 1
            cumulative += left_vol + right_vol

    value_area_low = float(buckets[low].left)
    value_area_high = float(buckets[high].right)
    return value_area_high, value_area_low
