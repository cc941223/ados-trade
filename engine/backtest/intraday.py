"""6.2 节 —— 日内短线场景成功判定：evaluate_intraday_signal()。

目标价 = 入场价 ± N×ATR（方向由信号自身的 direction 决定，N 默认 1.5）
止损价 = 入场价 ∓ M×ATR（M 默认 1.0）
判定：收盘前目标价先触及 -> win；止损价先触及 -> loss；都没触及 -> undecided
（不计入胜率分母，这是上层统计代码的事，本函数只负责标出 undecided）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import SignalOutcome


def evaluate_intraday_signal(
    entry_price: float,
    direction: str,
    atr: float,
    price_path: pd.DataFrame,
    target_multiplier: float = 1.5,
    stop_multiplier: float = 1.0,
    same_bar_tie_break: str = "stop_first",
) -> SignalOutcome:
    """判定一笔日内短线信号触发后到收盘的胜负。

    Parameters
    ----------
    entry_price : 入场价。
    direction : "bull" 或 "bear"，决定目标价/止损价在入场价的哪一侧。
    atr : 入场时的 ATR 值（例如 `engine.indicators.trend.atr` 算出的值）。
    price_path : 入场后到收盘的逐根 K 线数据，需包含 `high`/`low` 列，按
        时间升序排列，索引（DatetimeIndex 或序号）会被记到 `exit_ts`。
        不包含入场那一根本身（避免"入场瞬间就用当根 K 线的高低点判定"这种
        用到未来数据的问题——入场价通常就是那根 K 线收盘价附近，用同一根
        K 线的高低点去判断目标/止损是否触及，等于用了这根 K 线还没走完时
        不该知道的信息）。
    target_multiplier, stop_multiplier : 目标价/止损价的 ATR 倍数，默认
        1.5 / 1.0（对应规格书 6.2 节 N/M 的初始值）。

        ⚠️ V1 占位值，待 M3 回测校准：规格书原文写"N 初始取 1.5""M 初始取
        1"，措辞已经表明这是起始值，不是定论，跟本项目其它 V1 占位系数
        （`classify_regime` 的 `threshold=10.0` 等）性质相同，等回测积累
        数据后应该重新校准，不是从数据反推出来的最优值。
    same_bar_tie_break : 如果同一根 K 线内目标价和止损价同时被触及（K 线
        振幅足够大，高低点分别越过了两个阈值），无法从 OHLC 数据本身判断
        盘中实际触及顺序，这是所有基于 K 线（而非逐笔成交）做回测的通用
        局限。默认 `"stop_first"`——同一根 K 线内假定止损先触发（更保守，
        不会把无法确定的情况算成盈利，避免虚高胜率）。这是本函数的设计
        选择，不是规格书规定的规则，如果你更倾向"目标价先触发"或其它处理
        方式，传 `same_bar_tie_break="target_first"` 即可切换，或者告诉我
        改默认值。

    Returns
    -------
    SignalOutcome，`outcome` 为 "win"/"loss"/"undecided"，`exit_reason` 为
    "target_hit"/"stop_hit"/"undecided"。
    """
    if direction not in ("bull", "bear"):
        raise ValueError(f"direction 必须是 'bull' 或 'bear'，收到：{direction!r}")
    if same_bar_tie_break not in ("stop_first", "target_first"):
        raise ValueError("same_bar_tie_break 必须是 'stop_first' 或 'target_first'")

    if direction == "bull":
        target_price = entry_price + target_multiplier * atr
        stop_price = entry_price - stop_multiplier * atr
    else:
        target_price = entry_price - target_multiplier * atr
        stop_price = entry_price + stop_multiplier * atr

    for ts, bar in price_path.iterrows():
        high, low = bar["high"], bar["low"]

        if direction == "bull":
            target_hit = high >= target_price
            stop_hit = low <= stop_price
        else:
            target_hit = low <= target_price
            stop_hit = high >= stop_price

        if target_hit and stop_hit:
            first_hit = same_bar_tie_break  # "stop_first" -> 按止损处理，"target_first" -> 按目标处理
        elif target_hit:
            first_hit = "target_first"
        elif stop_hit:
            first_hit = "stop_first"
        else:
            continue

        if first_hit == "target_first":
            raw_return_pct = _signed_return(entry_price, target_price, direction)
            return SignalOutcome(
                outcome="win",
                exit_reason="target_hit",
                exit_price=target_price,
                exit_ts=ts,
                raw_return_pct=raw_return_pct,
            )
        else:
            raw_return_pct = _signed_return(entry_price, stop_price, direction)
            return SignalOutcome(
                outcome="loss",
                exit_reason="stop_hit",
                exit_price=stop_price,
                exit_ts=ts,
                raw_return_pct=raw_return_pct,
            )

    # 走完整个 price_path 都没触及任何一侧：未分出胜负。仍然算出"收盘时"的
    # 浮动收益率存进 raw_return_pct，方便留痕/事后复盘看走势，但 outcome
    # 明确标 undecided，上层统计代码应该据此把这笔信号排除在胜率分母外，
    # 不是本函数自己去决定"不算"就不产出任何数值。
    if len(price_path) > 0:
        last_close = price_path["close"].iloc[-1] if "close" in price_path.columns else None
        last_ts = price_path.index[-1]
    else:
        last_close = None
        last_ts = None

    raw_return_pct = _signed_return(entry_price, last_close, direction) if last_close is not None else None
    return SignalOutcome(
        outcome="undecided",
        exit_reason="undecided",
        exit_price=last_close,
        exit_ts=last_ts,
        raw_return_pct=raw_return_pct,
    )


def _signed_return(entry_price: float, exit_price: Optional[float], direction: str) -> Optional[float]:
    """按方向调整符号的收益率：bull 时价格涨是正收益，bear 时价格跌是正收益。"""
    if exit_price is None:
        return None
    raw = (exit_price - entry_price) / entry_price
    return raw if direction == "bull" else -raw
