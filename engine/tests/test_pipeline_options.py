"""run_options_pipeline() 端到端测试：模拟标的 K 线 + 多到期日期权链，
验证整条链路能跑通、字段对得上，以及 OpEx 权重覆盖确实生效。
"""
from datetime import date

import pytest

from engine.pipeline import run_options_pipeline
from engine.pipeline.base import PipelineResult
from engine.scoring.options import DEFAULT_WEIGHTS

from ._pipeline_fixtures import make_daily_ohlcv, make_option_chain


def test_options_pipeline_runs_end_to_end_and_fields_line_up():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365, 60 / 365, 90 / 365])

    result = run_options_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        underlying_ohlcv=underlying_ohlcv,
        option_chain=option_chain,
        target_expiry_years=30 / 365,
        iv_percentile=60.0,
    )

    assert isinstance(result, PipelineResult)
    assert result.scenario == "options"
    assert 0.0 <= result.bull_score <= 100.0
    assert 0.0 <= result.bear_score <= 100.0
    assert 0.0 <= result.confidence_score <= 100.0

    assert set(result.sub_scores) == {
        "iv_skew",
        "iv_term_structure",
        "max_pain_deviation",
        "iv_percentile",
        "earnings_surprise",
    }
    # 3 个到期日、每个到期日 5 个行使价，iv_skew/max_pain/iv_term_structure
    # 都应该能正常算出来（不缺失）；earnings_surprise 没在财报窗口期内，
    # 按 score_options 的规则视为不适用。
    assert result.sub_scores["iv_skew"].score is not None
    assert result.sub_scores["iv_term_structure"].score is not None
    assert result.sub_scores["max_pain_deviation"].score is not None
    assert result.sub_scores["iv_percentile"].score == 0.0  # 强制中性，不影响方向
    assert result.sub_scores["earnings_surprise"].score is None

    assert result.confidence_score == result.raw_confidence_score  # 没有事件触发


def test_options_pipeline_single_expiry_marks_term_structure_missing():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365])  # 只有一个到期日

    result = run_options_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        underlying_ohlcv=underlying_ohlcv,
        option_chain=option_chain,
        target_expiry_years=30 / 365,
    )

    assert result.sub_scores["iv_term_structure"].score is None
    # 其它子指标不受影响
    assert result.sub_scores["iv_skew"].score is not None
    assert result.sub_scores["max_pain_deviation"].score is not None


def test_options_pipeline_unknown_target_expiry_raises():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365])

    with pytest.raises(ValueError):
        run_options_pipeline(
            symbol="AAPL",
            current_date=date(2026, 8, 5),
            underlying_ohlcv=underlying_ohlcv,
            option_chain=option_chain,
            target_expiry_years=999 / 365,  # 链里不存在的到期日
        )


def test_options_pipeline_earnings_window_flag_and_confidence():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365, 60 / 365])

    result = run_options_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        underlying_ohlcv=underlying_ohlcv,
        option_chain=option_chain,
        target_expiry_years=30 / 365,
        earnings_surprise_avg=0.03,
        earnings_date=date(2026, 8, 6),  # 1 个交易日后，落在财报窗口期内
    )

    # 财报窗口期内，历史意外幅度子指标应该正常参与打分
    assert result.sub_scores["earnings_surprise"].score is not None
    # 财报事件同时触发 Confidence × 0.5
    assert result.confidence_score == result.raw_confidence_score * 0.5
    assert "财报临近，历史规律可能失效" in result.labels


def test_options_pipeline_opex_window_shifts_max_pain_weight():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365, 60 / 365])

    baseline = run_options_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        underlying_ohlcv=underlying_ohlcv,
        option_chain=option_chain,
        target_expiry_years=30 / 365,
    )
    with_opex = run_options_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        underlying_ohlcv=underlying_ohlcv,
        option_chain=option_chain,
        target_expiry_years=30 / 365,
        opex_date=date(2026, 8, 6),  # 1 个交易日后，落在默认 2 天窗口内
    )

    assert baseline.sub_scores["max_pain_deviation"].weight == DEFAULT_WEIGHTS["max_pain_deviation"]
    assert with_opex.sub_scores["max_pain_deviation"].weight == 0.35
    # 其余子指标权重应该等比例压缩，总和仍为 1
    total_weight = sum(sub.weight for sub in with_opex.sub_scores.values())
    assert abs(total_weight - 1.0) < 1e-9
    # 除 max_pain_deviation 外，压缩比例应该一致
    scale = (1.0 - 0.35) / (1.0 - DEFAULT_WEIGHTS["max_pain_deviation"])
    for key, default_weight in DEFAULT_WEIGHTS.items():
        if key == "max_pain_deviation":
            continue
        assert abs(with_opex.sub_scores[key].weight - default_weight * scale) < 1e-9


def test_options_pipeline_rejects_explicit_weights_in_score_kwargs():
    underlying_ohlcv = make_daily_ohlcv(n=30, start_price=100.0, seed=3)
    spot = float(underlying_ohlcv["close"].iloc[-1])
    option_chain = make_option_chain(spot, expiries_years=[30 / 365])

    with pytest.raises(ValueError):
        run_options_pipeline(
            symbol="AAPL",
            current_date=date(2026, 8, 5),
            underlying_ohlcv=underlying_ohlcv,
            option_chain=option_chain,
            target_expiry_years=30 / 365,
            score_kwargs={"weights": {"iv_skew": 0.5}},
        )
