"""日内短线场景的端到端编排：run_intraday_pipeline()。

原始数据 -> 指标 -> 打分 -> 事件调整 -> 最终结果，链路：

    ohlcv(DataFrame)
        --(engine.indicators.base.vwap / trend.atr / trend.sma)-->
        vwap / atr / 短&长期均线 / 量比
        --(engine.scoring.intraday.score_intraday)--> ScoreResult
        --(engine.events.adjustments.apply_event_adjustments)--> EventAdjustment
        --(engine.pipeline.base.combine_with_event_adjustment)--> PipelineResult

⚠️ 已知缺口：RSI。`score_intraday` 需要一个 0~100 的 RSI 值，但
`engine.indicators` 里目前没有任何函数计算 RSI——不是疏漏，是刻意没做：
RSI 有实质性的公式选择（Wilder 原始平滑 vs 简单移动平均的 gain/loss），
属于需要专门实现、写测试、跟规格书核对的指标计算，不是编排层可以顺手写一行
搞定的东西（这正是这一层"不应该引入新指标计算"的边界）。所以这里把 `rsi`
做成一个直接传入的参数——不提供时（默认 `None`）对应子指标在 `score_intraday`
里会被判定为缺失，正常参与完整度计算但不参与打分，不会报错。等
`engine.indicators.trend` 补上 RSI 之后，再把这里换成调用它。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from engine.events.adjustments import apply_event_adjustments
from engine.indicators.base import vwap
from engine.indicators.trend import atr, sma
from engine.scoring.intraday import score_intraday

from .base import PipelineResult, combine_with_event_adjustment, none_if_nan


def run_intraday_pipeline(
    symbol: str,
    current_date: date,
    ohlcv: pd.DataFrame,
    rsi: Optional[float] = None,
    short_ma_period: int = 5,
    long_ma_period: int = 20,
    atr_period: int = 14,
    volume_avg_period: int = 20,
    earnings_date: Optional[date] = None,
    earnings_confirmed: bool = True,
    macro_event_date: Optional[date] = None,
    score_kwargs: Optional[dict] = None,
    event_kwargs: Optional[dict] = None,
) -> PipelineResult:
    """日内短线场景端到端编排。

    Parameters
    ----------
    symbol : 标的代码，仅用于事件调整的审计留痕。
    current_date : 当前日期。
    ohlcv : 当日的 K 线数据，需包含 `open`/`high`/`low`/`close`/`volume` 列，
        按时间升序排列。**这应该是当日从开盘到当前时刻的分钟/tick 级数据**，
        不是多日日线——VWAP 的定义是"每个交易日从开盘归零重新累计"
        （见 `engine.indicators.base.vwap` 文档），传多日数据会把 VWAP 算错
        （累计到跨天，而不是当日）。均线/ATR/量比这几个子指标则需要足够长的
        历史窗口才有意义，如果 `ohlcv` 只包含当天的分钟线，这几个子指标会
        因为窗口不足而缺失（`sma`/`atr` 在窗口不足时返回 NaN，本函数会转成
        `None`，不会报错，只是这几个子指标不参与打分）——如果要让这几个
        子指标正常工作，`ohlcv` 需要包含足够多个交易日的收盘价历史。
    rsi : 见模块文档"已知缺口"——RSI 尚未在 engine.indicators 实现，这里是
        直接传入的外部值，不提供则该子指标视为缺失。
    short_ma_period, long_ma_period : 均线排列子指标用的短/长周期，默认
        5 日 / 20 日。
    atr_period : ATR 计算周期，默认 14（`engine.indicators.trend.atr` 的默认值）。
    volume_avg_period : 量比子指标的历史均量回看窗口，默认 20 日。
    earnings_date, earnings_confirmed, macro_event_date : 转发给
        `apply_event_adjustments` 的事件日期信息。
    score_kwargs : 转发给 `score_intraday` 的额外关键字参数（例如
        `weights` 覆盖权重表，或 `vwap_scale`/`atr_z_scale` 等饱和阈值覆盖），
        不在这里逐个列出每个 V1 占位参数，是为了不让编排层的签名膨胀到跟
        `score_intraday` 一模一样——需要覆盖时直接传对应参数名即可。
    event_kwargs : 转发给 `apply_event_adjustments` 的额外关键字参数
        （例如各事件的窗口天数/调整系数覆盖），同上不逐个列出。

    Returns
    -------
    PipelineResult
    """
    score_kwargs = dict(score_kwargs or {})
    event_kwargs = dict(event_kwargs or {})

    # ---- 1. 从原始 OHLCV 数据算指标（全部调用 engine.indicators 已有函数） ----
    price = float(ohlcv["close"].iloc[-1])
    reference_price = float(ohlcv["open"].iloc[-1])

    vwap_value = none_if_nan(float(vwap(ohlcv).iloc[-1]))
    atr_value = none_if_nan(float(atr(ohlcv, period=atr_period).iloc[-1]))
    short_ma = none_if_nan(float(sma(ohlcv["close"], short_ma_period).iloc[-1]))
    long_ma = none_if_nan(float(sma(ohlcv["close"], long_ma_period).iloc[-1]))

    # 量比 = 当前量 / 历史均量；均量用 sma() 算成交量的滑动平均。
    current_volume = float(ohlcv["volume"].iloc[-1])
    avg_volume = none_if_nan(float(sma(ohlcv["volume"], volume_avg_period).iloc[-1]))
    volume_ratio = current_volume / avg_volume if avg_volume else None

    # 当日涨跌幅：判断"放量"该加强哪个方向要用到，纯粹的 pct_change，不是
    # 指标计算。
    price_change_pct = none_if_nan(float(ohlcv["close"].pct_change().iloc[-1]))

    # ---- 2. 组装参数，调用 score_intraday ----
    score_result = score_intraday(
        price=price,
        vwap=vwap_value,
        rsi=rsi,
        volume_ratio=volume_ratio,
        price_change_pct=price_change_pct,
        atr=atr_value,
        reference_price=reference_price,
        short_ma=short_ma,
        long_ma=long_ma,
        **score_kwargs,
    )

    # ---- 3. 事件调整 ----
    event_adjustment = apply_event_adjustments(
        scenario="intraday",
        current_date=current_date,
        symbol=symbol,
        earnings_date=earnings_date,
        earnings_confirmed=earnings_confirmed,
        macro_event_date=macro_event_date,
        **event_kwargs,
    )

    # ---- 4. 合并输出 ----
    return combine_with_event_adjustment("intraday", score_result, event_adjustment)
