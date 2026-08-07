"""杠杆 ETF 对冲场景的端到端编排：run_leveraged_etf_pipeline()。

    etf_ohlcv + underlying_ohlcv(DataFrame)
        --(engine.indicators.trend.sma,
           engine.indicators.leveraged.volatility_decay_regression/
           vix_term_structure_ratio)-->
        200 日均线位置 / 波动损耗率 / VIX 期限结构
        --(engine.scoring.leveraged_etf.score_leveraged_etf)--> ScoreResult
        --(engine.events.adjustments.apply_event_adjustments)--> EventAdjustment
        --(engine.pipeline.base.combine_with_event_adjustment)--> PipelineResult

⚠️ 需要显式对齐的地方：`volatility_decay_regression` 要求 `etf_returns`/
`underlying_returns` 两条收益率序列按同一批交易日一一对应（长度相等、
顺序一致），但 `etf_ohlcv`（杠杆 ETF 自身）和 `underlying_ohlcv`（
broad_index 时是跟踪的指数，single_stock 时是标的个股）是两条独立的
K 线序列，行数、缺失日期都可能不一样（比如个股某天临时停牌，指数照常有
数据）。本函数用 `pandas` 按索引做 inner join 对齐两条收益率序列——这不是
新指标计算，是让两条序列在送进回归之前具备"同一天对同一天"的前提，回归
本身的数学逻辑完全没碰。**要求调用方传入的两个 DataFrame 使用能表达"哪天
对哪天"的索引**（例如 `DatetimeIndex`，或者两者本来就是按完全相同的交易日
顺序构造的 `RangeIndex`）——如果索引压根不对应真实交易日（比如两个
DataFrame 各自从 0 开始编号但实际交易日不同），inner join 不会报错但结果
是错的，这一点本函数无法从数据本身验证出来，需要调用方保证。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from engine.events.adjustments import apply_event_adjustments
from engine.indicators.leveraged import volatility_decay_regression, vix_term_structure_ratio
from engine.indicators.trend import sma
from engine.scoring.leveraged_etf import score_leveraged_etf

from .base import PipelineResult, combine_with_event_adjustment, none_if_nan


def run_leveraged_etf_pipeline(
    symbol: str,
    current_date: date,
    target_type: str,
    etf_ohlcv: pd.DataFrame,
    underlying_ohlcv: pd.DataFrame,
    leverage: float,
    vix_level: Optional[float] = None,
    front_vix: Optional[float] = None,
    back_vix: Optional[float] = None,
    decay_window_days: int = 30,
    ma200_period: int = 200,
    rebalance_date: Optional[date] = None,
    earnings_date: Optional[date] = None,
    earnings_confirmed: bool = True,
    underlying_symbol: Optional[str] = None,
    macro_event_date: Optional[date] = None,
    score_kwargs: Optional[dict] = None,
    event_kwargs: Optional[dict] = None,
) -> PipelineResult:
    """杠杆 ETF 对冲场景端到端编排。

    Parameters
    ----------
    symbol : 杠杆 ETF 自身的代码（例如 "TQQQ"/"TSLL"），转发给事件调整的
        `symbol` 参数。
    current_date : 当前日期。
    target_type : "broad_index" 或 "single_stock"，决定 `underlying_ohlcv`
        该传什么、以及财报/再平衡两条事件规则是否适用（转发给
        `score_leveraged_etf`/`apply_event_adjustments` 的同名参数）。
    etf_ohlcv : 杠杆 ETF 自身的日线数据，用于算它自己的日收益率（波动损耗率
        回归的 `etf_returns`）。
    underlying_ohlcv : 跟踪标的自身的日线数据——`target_type="broad_index"`
        时传跟踪的宽基指数（如 QQQ），`target_type="single_stock"` 时传
        标的个股（如 TSLA），不是"大盘"，见 `score_leveraged_etf` 文档里
        "标的类型区分"的说明。同时用于算 200 日均线位置。
    leverage : 该 ETF 的名义杠杆倍数（例如 TQQQ 传 3.0，SQQQ 传 -3.0）。
    vix_level : VIX 绝对水平（外部 VIX 指数数据，不是从 OHLCV 算出来的）。
    front_vix, back_vix : 用于算 VIX 期限结构比值的近月/远月读数（例如
        VIX9D/VIX，或 VIX/VIX3M）。两者任一缺失时，VIX 期限结构子指标视为
        缺失（不报错）。
    decay_window_days : 波动损耗率回归使用的对齐后交易日窗口，默认 30
        （对应规格书"近 30 天实际值"）。
    ma200_period : 200 日均线周期，默认 200。
    rebalance_date, earnings_date, earnings_confirmed, underlying_symbol,
    macro_event_date : 转发给 `apply_event_adjustments` 的事件日期信息。
    score_kwargs : 转发给 `score_leveraged_etf` 的额外关键字参数。
    event_kwargs : 转发给 `apply_event_adjustments` 的额外关键字参数。

    Returns
    -------
    PipelineResult，`regime` 字段会带上 "bull"/"chop"/"bear" 三态标签。
    """
    score_kwargs = dict(score_kwargs or {})
    event_kwargs = dict(event_kwargs or {})

    # ---- 1. 从原始 OHLCV 数据算指标 ----
    underlying_price = float(underlying_ohlcv["close"].iloc[-1])
    underlying_ma200 = none_if_nan(float(sma(underlying_ohlcv["close"], ma200_period).iloc[-1]))

    # 波动损耗率回归：先各自算日收益率，再按索引 inner join 对齐（见模块
    # 文档的对齐说明），最后只取最近 decay_window_days 个对齐后的交易日。
    etf_returns_full = etf_ohlcv["close"].pct_change()
    underlying_returns_full = underlying_ohlcv["close"].pct_change()
    aligned = pd.concat(
        [etf_returns_full.rename("etf"), underlying_returns_full.rename("underlying")], axis=1, join="inner"
    ).dropna()
    aligned = aligned.tail(decay_window_days)

    daily_decay = None
    if len(aligned) >= 2:
        decay_result = volatility_decay_regression(
            aligned["etf"].tolist(), aligned["underlying"].tolist(), leverage
        )
        daily_decay = decay_result.daily_decay
    # 对齐后不足 2 个交易日：波动损耗率子指标视为缺失，不强行报错——数据
    # 刚接入、历史窗口还没积累够时，这是预期会发生的情况。

    term_structure_ratio = None
    if front_vix is not None and back_vix is not None:
        term_structure_ratio = vix_term_structure_ratio(front_vix, back_vix)["ratio"]

    # ---- 2. 组装参数，调用 score_leveraged_etf ----
    score_result = score_leveraged_etf(
        target_type=target_type,
        underlying_price=underlying_price,
        vix=vix_level,
        vix_term_structure_ratio=term_structure_ratio,
        underlying_ma200=underlying_ma200,
        daily_decay=none_if_nan(daily_decay),
        **score_kwargs,
    )

    # ---- 3. 事件调整 ----
    event_adjustment = apply_event_adjustments(
        scenario="leveraged_etf",
        current_date=current_date,
        target_type=target_type,
        symbol=symbol,
        underlying_symbol=underlying_symbol,
        earnings_date=earnings_date,
        earnings_confirmed=earnings_confirmed,
        rebalance_date=rebalance_date,
        macro_event_date=macro_event_date,
        **event_kwargs,
    )

    # ---- 4. 合并输出 ----
    return combine_with_event_adjustment("leveraged_etf", score_result, event_adjustment)
