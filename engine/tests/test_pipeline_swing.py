"""run_swing_pipeline() 端到端测试：模拟标的+基准日线数据，验证整条链路
能跑通、字段对得上。
"""
from datetime import date

from engine.pipeline import run_swing_pipeline
from engine.pipeline.base import PipelineResult

from ._pipeline_fixtures import make_daily_ohlcv


def test_swing_pipeline_runs_end_to_end_and_fields_line_up():
    # 标的比基准涨得更快，制造一个"跑赢大盘"的场景
    asset_ohlcv = make_daily_ohlcv(n=260, start_price=100.0, daily_drift=0.15, daily_vol=1.0, seed=1)
    benchmark_ohlcv = make_daily_ohlcv(n=260, start_price=100.0, daily_drift=0.03, daily_vol=0.8, seed=2)

    result = run_swing_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        asset_ohlcv=asset_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
        put_volume=8000,
        call_volume=10000,
        va_bins=10,
        poc_bins=10,
    )

    assert isinstance(result, PipelineResult)
    assert result.scenario == "swing"
    assert 0.0 <= result.bull_score <= 100.0
    assert 0.0 <= result.bear_score <= 100.0
    assert 0.0 <= result.confidence_score <= 100.0

    assert set(result.sub_scores) == {
        "relative_strength",
        "ma_trend",
        "volume_profile_position",
        "put_call_ratio",
        "sector_alignment",
    }
    # 260 个交易日足够覆盖默认 rs_period=200/ma200_period=200 的窗口
    for key in result.sub_scores:
        assert result.sub_scores[key].score is not None, f"{key} 不应该缺失"

    assert result.confidence_score == result.raw_confidence_score  # 没有事件触发
    assert result.labels == []
    assert result.independent_warnings == []


def test_swing_pipeline_insufficient_history_marks_rs_and_ma_missing():
    # 只给 30 个交易日，不够默认 rs_period=200/ma200_period=200 的窗口
    asset_ohlcv = make_daily_ohlcv(n=30, seed=1)
    benchmark_ohlcv = make_daily_ohlcv(n=30, seed=2)

    result = run_swing_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        asset_ohlcv=asset_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
    )

    assert result.sub_scores["relative_strength"].score is None
    assert result.sub_scores["ma_trend"].score is None  # ma50/ma200 都不够窗口
    # Volume Profile 位置不依赖长窗口，应该正常有值
    assert result.sub_scores["volume_profile_position"].score is not None
    assert 0.0 <= result.confidence_score <= 100.0


def test_swing_pipeline_without_put_call_volume_marks_pcr_missing():
    asset_ohlcv = make_daily_ohlcv(n=260, seed=1)
    benchmark_ohlcv = make_daily_ohlcv(n=260, seed=2)

    result = run_swing_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        asset_ohlcv=asset_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
    )
    assert result.sub_scores["put_call_ratio"].score is None


def test_swing_pipeline_earnings_event_lowers_confidence():
    asset_ohlcv = make_daily_ohlcv(n=260, seed=1)
    benchmark_ohlcv = make_daily_ohlcv(n=260, seed=2)

    baseline = run_swing_pipeline(
        symbol="AAPL", current_date=date(2026, 8, 5), asset_ohlcv=asset_ohlcv, benchmark_ohlcv=benchmark_ohlcv
    )
    with_earnings = run_swing_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        asset_ohlcv=asset_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
        earnings_date=date(2026, 8, 6),  # 1 个交易日后
    )

    assert with_earnings.confidence_score == baseline.raw_confidence_score * 0.5
    assert "财报临近，历史规律可能失效" in with_earnings.labels
