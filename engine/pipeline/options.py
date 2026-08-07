"""期权策略场景的端到端编排：run_options_pipeline()。

    underlying_ohlcv + option_chain(DataFrame)
        --(engine.indicators.options.iv_skew/max_pain/iv_term_structure)-->
        IV Skew / Max Pain / IV Term Structure
        --(engine.scoring.options.score_options)--> ScoreResult
        --(engine.events.adjustments.apply_event_adjustments)--> EventAdjustment
        --(engine.pipeline.base.combine_with_event_adjustment)--> PipelineResult

期权链数据结构（`option_chain` 参数）：一个"长表"DataFrame，一行对应一个
(到期日, 行使价) 组合，需要包含列：
    expiry_years : float，该到期日的年化剩余期限（例如 30 天 = 30/365），
        跟 `engine.indicators.options.iv_skew`/`iv_term_structure` 的 `T`/
        `expiries_T` 参数是同一个单位，本函数不做"日期转年化期限"的计算——
        这一步理论上只是简单除法，但涉及"用日历日还是交易日"这类需要业务
        约定的选择，交给数据接入层（`data/`）在写入时就转换好，编排层直接
        消费现成的年化数字。
    strike, iv, call_oi, put_oi : 该行使价在该到期日的隐含波动率与
        Call/Put 未平仓量。

⚠️ 两处需要显式选择、本函数不代为决定的地方：
1. `target_expiry_years`：IV Skew 和 Max Pain 都是"同一到期日"内的计算
   （`iv_skew`/`max_pain` 的入参都只接受单一到期日的 strikes/ivs/OI），
   但一条标的的期权链通常有多个到期日。选哪个到期日代表"当前的期权策略
   评分"是一个业务判断（例如"选最近的月度到期日"还是"选最接近某个目标
   DTE 的到期日"），不是数据整理，所以要求调用方显式传入
   `target_expiry_years`，本函数只按这个值从 `option_chain` 里筛选对应的
   行，不做任何"自动挑一个看起来合理的到期日"的隐式判断。
2. IV Term Structure 需要"每个到期日的 ATM IV"（`iv_term_structure` 的
   `atm_ivs` 参数），从长表里为每个到期日选出"最接近 spot 的行使价对应的
   IV"是一个选择型操作（取最小绝对差值，没有计算公式，跟 `poc()`/
   `value_area()` 内部挑桶是同一类操作），本函数在 `_atm_iv_per_expiry()`
   里实现，属于"数据整理"而非"新指标计算"。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from engine.events.adjustments import apply_event_adjustments, trading_days_between
from engine.indicators.options import iv_skew, iv_term_structure, max_pain
from engine.scoring.options import DEFAULT_WEIGHTS as SCORE_OPTIONS_DEFAULT_WEIGHTS
from engine.scoring.options import score_options

from .base import PipelineResult, combine_with_event_adjustment, none_if_nan, redistribute_weights_for_override


def _atm_iv_per_expiry(option_chain: pd.DataFrame, spot: float) -> tuple[list[float], list[float]]:
    """按到期日分组，每组取"离 spot 最近的行使价"对应的 IV，作为该到期日的
    ATM IV。这是数据整理（选择最近的一行），不是新指标计算——公式判断全部
    留给 `engine.indicators.options.iv_term_structure`，这里只负责把长表
    "压"成它要的 (expiries_T, atm_ivs) 两个列表。
    """
    expiries_T: list[float] = []
    atm_ivs: list[float] = []
    for expiry_years, group in option_chain.groupby("expiry_years"):
        idx = (group["strike"] - spot).abs().idxmin()
        expiries_T.append(float(expiry_years))
        atm_ivs.append(float(group.loc[idx, "iv"]))
    return expiries_T, atm_ivs


def run_options_pipeline(
    symbol: str,
    current_date: date,
    underlying_ohlcv: pd.DataFrame,
    option_chain: pd.DataFrame,
    target_expiry_years: float,
    risk_free_rate: float = 0.0,
    iv_percentile: Optional[float] = None,
    earnings_surprise_avg: Optional[float] = None,
    earnings_window_days: int = 3,
    earnings_date: Optional[date] = None,
    earnings_confirmed: bool = True,
    opex_date: Optional[date] = None,
    is_triple_witching: bool = False,
    ex_dividend_date: Optional[date] = None,
    holds_deep_itm_covered_call: bool = False,
    macro_event_date: Optional[date] = None,
    score_kwargs: Optional[dict] = None,
    event_kwargs: Optional[dict] = None,
) -> PipelineResult:
    """期权策略场景端到端编排。

    Parameters
    ----------
    symbol : 标的代码。
    current_date : 当前日期。
    underlying_ohlcv : 标的自身的日线 K 线数据，用于取当前价格（`close`
        最后一行）。
    option_chain : 期权链长表，见模块文档里的结构说明。
    target_expiry_years : IV Skew / Max Pain 使用哪个到期日，见模块文档
        "已知需要显式选择"的第 1 点。
    risk_free_rate : 传给 `iv_skew`（进而传给 Black-Scholes）的无风险利率。
    iv_percentile : IV 百分位（0~100），外部数据源直接给出的字段（IBKR
        `implied_volatility_percentile`），不是本函数计算出来的。
    earnings_surprise_avg : 历史财报意外幅度均值，外部数据源（Finnhub）
        直接给出的字段。
    earnings_window_days : 判定"当前是否处于财报窗口期"（`score_options`
        的 `in_earnings_window` 参数）用的交易日窗口，同时也会作为
        `apply_event_adjustments` 的 `earnings_window_days` 传入——两处用
        同一个数字，避免"财报窗口"在打分层和事件调整层各自定义出不一致的
        窗口。
    earnings_date, earnings_confirmed, opex_date, is_triple_witching,
    ex_dividend_date, holds_deep_itm_covered_call, macro_event_date :
        转发给 `apply_event_adjustments` 的事件日期信息。
    score_kwargs : 转发给 `score_options` 的额外关键字参数。**不要在这里传
        `weights`**——如果 OpEx 规则触发了 Max Pain 权重调整，本函数会自己
        算出覆盖后的权重表并传给 `score_options`，跟 `score_kwargs` 里的
        `weights` 会冲突（如果你确实需要额外覆盖其它子指标的权重，请通过
        `event_kwargs` 覆盖 `opex_max_pain_weight` 等系数，而不是在
        `score_kwargs` 里直接塞 `weights`）。
    event_kwargs : 转发给 `apply_event_adjustments` 的额外关键字参数。

    Returns
    -------
    PipelineResult
    """
    score_kwargs = dict(score_kwargs or {})
    event_kwargs = dict(event_kwargs or {})
    if "weights" in score_kwargs:
        raise ValueError(
            "score_kwargs 不支持传 'weights'——Max Pain 权重覆盖由本函数根据 OpEx "
            "事件规则自动计算，直接传 weights 会覆盖掉这个逻辑，请通过 event_kwargs "
            "调整 opex_max_pain_weight 等系数。"
        )

    # ---- 1. 从原始数据算指标 ----
    spot = float(underlying_ohlcv["close"].iloc[-1])

    expiry_chain = option_chain[option_chain["expiry_years"] == target_expiry_years]
    if expiry_chain.empty:
        raise ValueError(f"option_chain 里没有 expiry_years={target_expiry_years} 对应的行")

    strikes = expiry_chain["strike"].tolist()
    ivs = expiry_chain["iv"].tolist()
    call_oi = expiry_chain["call_oi"].tolist()
    put_oi = expiry_chain["put_oi"].tolist()

    skew_result = iv_skew(strikes, ivs, spot, target_expiry_years, risk_free_rate)
    iv_skew_value = skew_result["skew"]

    max_pain_value = max_pain(strikes, call_oi, put_oi)

    expiries_T, atm_ivs = _atm_iv_per_expiry(option_chain, spot)
    if len(expiries_T) >= 2:
        term_structure_result = iv_term_structure(expiries_T, atm_ivs)
        iv_term_structure_ratio = term_structure_result["near_far_ratio"]
    else:
        # 只有一个到期日时算不出期限结构（iv_term_structure 要求至少两个
        # 到期日），该子指标视为缺失，不是错误——期权链本来就可能只覆盖了
        # 一个到期日（比如刚接入数据源时）。
        iv_term_structure_ratio = None

    in_earnings_window = (
        earnings_date is not None and 0 < trading_days_between(current_date, earnings_date) <= earnings_window_days
    )

    # ---- 2. 事件调整（先算，因为 Max Pain 权重覆盖需要在调用 score_options 前算好） ----
    event_adjustment = apply_event_adjustments(
        scenario="options",
        current_date=current_date,
        symbol=symbol,
        earnings_date=earnings_date,
        earnings_confirmed=earnings_confirmed,
        earnings_window_days=earnings_window_days,
        opex_date=opex_date,
        is_triple_witching=is_triple_witching,
        ex_dividend_date=ex_dividend_date,
        holds_deep_itm_covered_call=holds_deep_itm_covered_call,
        macro_event_date=macro_event_date,
        **event_kwargs,
    )

    weights = redistribute_weights_for_override(
        SCORE_OPTIONS_DEFAULT_WEIGHTS, "max_pain_deviation", event_adjustment.max_pain_weight
    )
    if weights is not None:
        score_kwargs["weights"] = weights

    # ---- 3. 组装参数，调用 score_options ----
    score_result = score_options(
        price=spot,
        iv_skew=none_if_nan(iv_skew_value),
        iv_term_structure_ratio=none_if_nan(iv_term_structure_ratio),
        max_pain=none_if_nan(max_pain_value),
        iv_percentile=iv_percentile,
        earnings_surprise_avg=earnings_surprise_avg,
        in_earnings_window=in_earnings_window,
        **score_kwargs,
    )

    # ---- 4. 合并输出 ----
    return combine_with_event_adjustment("options", score_result, event_adjustment)
