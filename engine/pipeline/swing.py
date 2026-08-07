"""波段 / 趋势跟踪场景的端到端编排：run_swing_pipeline()。

    asset_ohlcv/benchmark_ohlcv(DataFrame)
        --(engine.indicators.trend.relative_strength/sma,
           engine.indicators.base.value_area/poc,
           engine.indicators.options.put_call_ratio)-->
        RS / 均线 / Value Area / POC / PCR / 板块联动
        --(engine.scoring.swing.score_swing)--> ScoreResult
        --(engine.events.adjustments.apply_event_adjustments)--> EventAdjustment
        --(engine.pipeline.base.combine_with_event_adjustment)--> PipelineResult

⚠️ Put/Call Ratio 的原始输入（当日 Put/Call 成交量）不在 `asset_ohlcv`
（正股 K 线）里，属于期权链数据，本函数按参数直接接收 `put_volume`/
`call_volume` 两个标量，不在这里假装能从股票 OHLCV 里推出来。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from engine.events.adjustments import apply_event_adjustments
from engine.indicators.base import poc, value_area
from engine.indicators.options import put_call_ratio
from engine.indicators.trend import relative_strength, sma
from engine.scoring.swing import score_swing

from .base import PipelineResult, combine_with_event_adjustment, none_if_nan


def run_swing_pipeline(
    symbol: str,
    current_date: date,
    asset_ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame,
    put_volume: Optional[float] = None,
    call_volume: Optional[float] = None,
    rs_period: int = 200,
    ma50_period: int = 50,
    ma200_period: int = 200,
    va_bins: int | list[float] = 10,
    va_percentage: float = 0.7,
    poc_bins: int | list[float] = 10,
    earnings_date: Optional[date] = None,
    earnings_confirmed: bool = True,
    macro_event_date: Optional[date] = None,
    score_kwargs: Optional[dict] = None,
    event_kwargs: Optional[dict] = None,
) -> PipelineResult:
    """波段 / 趋势跟踪场景端到端编排。

    Parameters
    ----------
    symbol : 标的代码，仅用于事件调整的审计留痕。
    current_date : 当前日期。
    asset_ohlcv : 标的自身的日线 K 线数据（`open`/`high`/`low`/`close`/
        `volume`），按时间升序排列，需要覆盖至少 `rs_period`/`ma200_period`
        （默认都是 200）个交易日，否则相对强度、200 日均线会因为窗口不足
        缺失（转成 `None`，不会报错）。
    benchmark_ohlcv : 大盘或板块 ETF（例如 QQQ）的日线 K 线数据，跟
        `asset_ohlcv` 同样的列结构，用于算相对强度 RS 和"板块/大盘联动"。
        `engine.indicators.trend.relative_strength` 要求两条序列按索引对齐，
        这里假定调用方传入的两个 DataFrame 已经是按同一组交易日对齐的
        （行数、顺序一致）——如果两者取数来源不同步（比如一个有假期数据
        一个没有），需要调用方自己先对齐，本函数不做隐式的日期对齐/插值，
        避免在编排层里悄悄引入"缺数据怎么补"这类需要业务判断的逻辑。
    put_volume, call_volume : 当日 Put/Call 总成交量，用于算 Put/Call Ratio。
        这是期权链数据，不在正股 OHLCV 里，所以单独作为参数接收。
    rs_period : 传给 `relative_strength` 的 Mansfield RS 均线周期，默认 200。
    ma50_period, ma200_period : 中长期均线周期，默认 50/200。
    va_bins, va_percentage : 传给 `value_area` 的分桶数（或显式桶边界）与
        目标覆盖比例。
    poc_bins : 传给 `poc` 的分桶数（或显式桶边界）。
    earnings_date, earnings_confirmed, macro_event_date : 转发给
        `apply_event_adjustments` 的事件日期信息（波段场景不涉及 OpEx/除息/
        再平衡规则，所以这里没有对应参数）。
    score_kwargs : 转发给 `score_swing` 的额外关键字参数（权重表覆盖、
        饱和阈值覆盖等）。
    event_kwargs : 转发给 `apply_event_adjustments` 的额外关键字参数。

    Returns
    -------
    PipelineResult
    """
    score_kwargs = dict(score_kwargs or {})
    event_kwargs = dict(event_kwargs or {})

    # ---- 1. 从原始 OHLCV 数据算指标 ----
    price = float(asset_ohlcv["close"].iloc[-1])

    rs_series = relative_strength(asset_ohlcv["close"], benchmark_ohlcv["close"], period=rs_period)
    rs_value = none_if_nan(float(rs_series.iloc[-1]))

    ma50 = none_if_nan(float(sma(asset_ohlcv["close"], ma50_period).iloc[-1]))
    ma200 = none_if_nan(float(sma(asset_ohlcv["close"], ma200_period).iloc[-1]))

    value_area_high, value_area_low = value_area(asset_ohlcv, bins=va_bins, percentage=va_percentage)
    value_area_high = none_if_nan(value_area_high)
    value_area_low = none_if_nan(value_area_low)
    poc_value = none_if_nan(poc(asset_ohlcv, bins=poc_bins))

    pcr_value = (
        put_call_ratio(put_volume, call_volume) if put_volume is not None and call_volume is not None else None
    )
    pcr_value = none_if_nan(pcr_value)

    # 板块/大盘联动：用基准同期涨跌幅本身作为方向/力度参照，纯 pct_change。
    benchmark_return = none_if_nan(float(benchmark_ohlcv["close"].pct_change().iloc[-1]))

    # ---- 2. 组装参数，调用 score_swing ----
    score_result = score_swing(
        price=price,
        rs_value=rs_value,
        ma50=ma50,
        ma200=ma200,
        value_area_high=value_area_high,
        value_area_low=value_area_low,
        poc=poc_value,
        put_call_ratio=pcr_value,
        benchmark_return=benchmark_return,
        **score_kwargs,
    )

    # ---- 3. 事件调整 ----
    event_adjustment = apply_event_adjustments(
        scenario="swing",
        current_date=current_date,
        symbol=symbol,
        earnings_date=earnings_date,
        earnings_confirmed=earnings_confirmed,
        macro_event_date=macro_event_date,
        **event_kwargs,
    )

    # ---- 4. 合并输出 ----
    return combine_with_event_adjustment("swing", score_result, event_adjustment)
