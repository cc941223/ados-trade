"""engine/pipeline 端到端测试的共用模拟数据构造工具（不是测试文件本身，
pytest 不会收集这个文件，命名以下划线开头就是为了避免被当成测试模块）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_daily_ohlcv(
    n: int = 260,
    start_price: float = 100.0,
    daily_drift: float = 0.0,
    daily_vol: float = 1.0,
    seed: int = 0,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """构造 n 个交易日的模拟日线 OHLCV（DatetimeIndex，工作日频率）。

    收盘价用随机游走生成（`daily_drift` 控制趋势，`daily_vol` 控制波动），
    high/low 在 close 基础上加一个小的固定比例扰动，open 用前一日 close
    （首日用 close 自身），volume 用固定基准值加一点随机扰动，足够覆盖
    `engine.indicators` 里各函数需要的列，不追求逼真，只追求"结构对、
    数值合理、可复现"。
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(daily_drift, daily_vol, n)
    close = start_price + np.cumsum(steps)
    close = np.maximum(close, 1.0)  # 避免随机游走走到负数

    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = rng.uniform(0.8, 1.2, n) * 1_000_000

    dates = pd.date_range(start_date, periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


def make_intraday_ohlcv(
    n: int = 60,
    start_price: float = 100.0,
    daily_drift: float = 0.02,
    daily_vol: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """构造 n 根分钟线的模拟当日 K 线（用于 VWAP 场景——VWAP 要求"当日从
    开盘归零累计"，所以不能直接拿多日日线代替）。"""
    rng = np.random.default_rng(seed)
    steps = rng.normal(daily_drift, daily_vol, n)
    close = start_price + np.cumsum(steps)
    close = np.maximum(close, 1.0)

    open_ = np.roll(close, 1)
    open_[0] = start_price

    high = np.maximum(open_, close) + rng.uniform(0, 0.05, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.05, n)
    volume = rng.uniform(0.8, 1.2, n) * 10_000

    index = pd.RangeIndex(n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def make_option_chain(spot: float, expiries_years: list[float]) -> pd.DataFrame:
    """构造一个覆盖多个到期日、每个到期日 5 个行使价的模拟期权链长表，
    行使价围绕 spot 展开，IV 带一个简单的微笑形态（越偏离 spot 越高），
    足够让 iv_skew/max_pain/iv_term_structure 都能正常算出非 NaN 的结果。
    """
    rows = []
    strikes = [spot * m for m in (0.9, 0.95, 1.0, 1.05, 1.1)]
    for T in expiries_years:
        base_iv = 0.20 + 0.05 * T  # 到期时间越长基础 IV 略高（简单 contango 形态）
        for strike in strikes:
            smile = 0.08 * abs(strike - spot) / spot
            iv = base_iv + smile
            moneyness = strike - spot
            call_oi = max(500 - 40 * abs(moneyness), 50)
            put_oi = max(500 - 40 * abs(moneyness), 50)
            rows.append(
                {
                    "expiry_years": T,
                    "strike": strike,
                    "iv": iv,
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                }
            )
    return pd.DataFrame(rows)
