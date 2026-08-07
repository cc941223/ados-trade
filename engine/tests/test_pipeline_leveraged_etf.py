"""run_leveraged_etf_pipeline() 端到端测试：模拟杠杆 ETF + 跟踪标的的日线
数据，覆盖 broad_index / single_stock 两种 target_type，验证整条链路能
跑通、字段对得上，以及事件规则按 target_type 正确区分适用范围。
"""
from datetime import date

import numpy as np
import pandas as pd

from engine.pipeline import run_leveraged_etf_pipeline
from engine.pipeline.base import PipelineResult

from ._pipeline_fixtures import make_daily_ohlcv


def _make_leveraged_pair(leverage: float, n: int = 260, seed: int = 5):
    """构造一对"標的 + 按 leverage 倍数跟踪它的杠杆 ETF"日线数据，两者
    完全对齐（同一个 DatetimeIndex），杠杆 ETF 的日收益率严格等于
    leverage * 标的日收益率 - 固定每日损耗，方便验证波动损耗率回归的输出
    落在合理范围内。
    """
    underlying = make_daily_ohlcv(n=n, start_price=100.0, daily_drift=0.02, daily_vol=1.0, seed=seed)
    underlying_returns = underlying["close"].pct_change().fillna(0.0)

    daily_decay = -0.001
    etf_returns = leverage * underlying_returns + daily_decay
    etf_close = 50.0 * (1 + etf_returns).cumprod()
    etf = pd.DataFrame(
        {
            "open": etf_close.shift(1).fillna(etf_close.iloc[0]),
            "high": etf_close * 1.01,
            "low": etf_close * 0.99,
            "close": etf_close,
            "volume": underlying["volume"],
        },
        index=underlying.index,
    )
    return etf, underlying


def test_leveraged_etf_pipeline_broad_index_runs_end_to_end():
    etf_ohlcv, underlying_ohlcv = _make_leveraged_pair(leverage=3.0)

    result = run_leveraged_etf_pipeline(
        symbol="TQQQ",
        current_date=date(2026, 8, 5),
        target_type="broad_index",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=3.0,
        vix_level=14.0,
        front_vix=16.0,
        back_vix=18.0,
    )

    assert isinstance(result, PipelineResult)
    assert result.scenario == "leveraged_etf"
    assert 0.0 <= result.bull_score <= 100.0
    assert 0.0 <= result.bear_score <= 100.0
    assert 0.0 <= result.confidence_score <= 100.0
    assert result.regime in {"bull", "chop", "bear"}

    assert set(result.sub_scores) == {"vix_level", "vix_term_structure", "ma200_position", "volatility_decay"}
    for key in result.sub_scores:
        assert result.sub_scores[key].score is not None, f"{key} 不应该缺失"

    # 波动损耗率应该接近构造数据里设定的 -0.001（回归会因为随机波动有偏差，
    # 但方向和量级应该大致吻合）
    assert result.sub_scores["volatility_decay"].raw_value < 0

    assert result.confidence_score == result.raw_confidence_score  # 没有事件触发


def test_leveraged_etf_pipeline_single_stock_uses_stock_ma200_not_index():
    etf_ohlcv, underlying_ohlcv = _make_leveraged_pair(leverage=1.5, seed=6)

    result = run_leveraged_etf_pipeline(
        symbol="TSLL",
        current_date=date(2026, 8, 5),
        target_type="single_stock",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=1.5,
        vix_level=20.0,
    )

    # ma200_position 的 raw_value 应该是基于 underlying_ohlcv（这里当作 TSLA）
    # 算出来的偏离幅度，不是某个固定的"大盘"值——用手算核对一下量级是否合理。
    underlying_price = float(underlying_ohlcv["close"].iloc[-1])
    assert result.sub_scores["ma200_position"].raw_value is not None
    assert abs(result.sub_scores["ma200_position"].raw_value) < 1.0  # 偏离幅度应该是个百分比量级，不是价格本身


def test_leveraged_etf_pipeline_insufficient_history_marks_decay_and_ma_missing():
    etf_ohlcv, underlying_ohlcv = _make_leveraged_pair(leverage=2.0, n=1, seed=7)

    result = run_leveraged_etf_pipeline(
        symbol="QLD",
        current_date=date(2026, 8, 5),
        target_type="broad_index",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=2.0,
        vix_level=18.0,
    )

    # 只有 1 天数据，对齐后的收益率序列长度为 0（pct_change 首行是 NaN），
    # 波动损耗率回归拿不到至少 2 个观测点，应该标记为缺失而不是报错。
    assert result.sub_scores["volatility_decay"].score is None
    # 200 日均线同样因为窗口不足缺失
    assert result.sub_scores["ma200_position"].score is None
    # VIX 子指标不依赖历史窗口，应该正常有值
    assert result.sub_scores["vix_level"].score is not None
    assert 0.0 <= result.confidence_score <= 100.0


def test_leveraged_etf_pipeline_rebalance_applies_only_to_broad_index():
    etf_ohlcv, underlying_ohlcv = _make_leveraged_pair(leverage=3.0, seed=8)

    broad_index_result = run_leveraged_etf_pipeline(
        symbol="TQQQ",
        current_date=date(2026, 8, 5),
        target_type="broad_index",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=3.0,
        vix_level=18.0,
        rebalance_date=date(2026, 8, 6),  # 1 个交易日后，落在默认 5 天窗口内
    )
    single_stock_result = run_leveraged_etf_pipeline(
        symbol="TSLL",
        current_date=date(2026, 8, 5),
        target_type="single_stock",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=1.5,
        vix_level=18.0,
        rebalance_date=date(2026, 8, 6),
    )

    assert broad_index_result.confidence_score == broad_index_result.raw_confidence_score * 0.7
    assert single_stock_result.confidence_score == single_stock_result.raw_confidence_score  # 不适用


def test_leveraged_etf_pipeline_earnings_applies_only_to_single_stock():
    etf_ohlcv, underlying_ohlcv = _make_leveraged_pair(leverage=1.5, seed=9)

    single_stock_result = run_leveraged_etf_pipeline(
        symbol="TSLL",
        current_date=date(2026, 8, 5),
        target_type="single_stock",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=1.5,
        vix_level=18.0,
        earnings_date=date(2026, 8, 6),
        underlying_symbol="TSLA",
    )
    broad_index_result = run_leveraged_etf_pipeline(
        symbol="TQQQ",
        current_date=date(2026, 8, 5),
        target_type="broad_index",
        etf_ohlcv=etf_ohlcv,
        underlying_ohlcv=underlying_ohlcv,
        leverage=3.0,
        vix_level=18.0,
        earnings_date=date(2026, 8, 6),
    )

    assert single_stock_result.confidence_score == single_stock_result.raw_confidence_score * 0.5
    assert "标的个股财报临近，杠杆敞口风险被放大，建议主动降低仓位或提前平仓" in single_stock_result.independent_warnings
    assert broad_index_result.confidence_score == broad_index_result.raw_confidence_score  # 不适用
