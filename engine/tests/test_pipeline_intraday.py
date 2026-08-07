"""run_intraday_pipeline() 端到端测试：模拟当日分钟线数据，验证从原始
OHLCV 到最终 Bull/Bear/Confidence Score 整条链路能跑通、字段对得上。
"""
from datetime import date

from engine.pipeline import run_intraday_pipeline
from engine.pipeline.base import PipelineResult

from ._pipeline_fixtures import make_intraday_ohlcv


def test_intraday_pipeline_runs_end_to_end_and_fields_line_up():
    ohlcv = make_intraday_ohlcv(n=60)

    result = run_intraday_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        ohlcv=ohlcv,
        rsi=45.0,
    )

    assert isinstance(result, PipelineResult)
    assert result.scenario == "intraday"
    assert 0.0 <= result.bull_score <= 100.0
    assert 0.0 <= result.bear_score <= 100.0
    assert 0.0 <= result.raw_confidence_score <= 100.0
    assert 0.0 <= result.confidence_score <= 100.0

    # 5 个子指标都应该出现在 sub_scores 里（即便部分因为窗口不足而缺失）
    assert set(result.sub_scores) == {"price_vs_vwap", "rsi", "volume_ratio", "atr_position", "ma_alignment"}

    # 60 根分钟线足够覆盖默认的 short_ma(5)/long_ma(20)/atr(14)/volume_avg(20)
    # 窗口，这几个子指标应该都有值（不是 None），rsi 是显式传入的也应该有值。
    for key in ("price_vs_vwap", "rsi", "volume_ratio", "atr_position", "ma_alignment"):
        assert result.sub_scores[key].score is not None, f"{key} 不应该缺失"

    # 没有触发任何事件时，confidence 不应该被调整
    assert result.confidence_score == result.raw_confidence_score
    assert result.labels == []
    assert result.independent_warnings == []


def test_intraday_pipeline_missing_rsi_is_gap_not_error():
    # RSI 尚未在 engine.indicators 实现，不传时应该被 score_intraday 当作
    # 缺失子指标处理，而不是报错。
    ohlcv = make_intraday_ohlcv(n=60)
    result = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv)

    assert result.sub_scores["rsi"].score is None
    assert result.sub_scores["rsi"].raw_value is None
    # 其它子指标不受影响，完整度应该小于 1（5 个里缺 1 个）
    assert result.sub_scores["price_vs_vwap"].score is not None


def test_intraday_pipeline_insufficient_history_marks_ma_missing_not_nan():
    # 只给 3 根 K 线，远不够默认 short_ma_period=5/long_ma_period=20/
    # atr_period=14/volume_avg_period=20 的窗口，这几个子指标应该被转成
    # None（缺失），而不是让 NaN 悄悄流进打分逻辑。
    ohlcv = make_intraday_ohlcv(n=3)
    result = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv, rsi=50.0)

    assert result.sub_scores["ma_alignment"].score is None
    assert result.sub_scores["atr_position"].score is None
    assert result.sub_scores["volume_ratio"].score is None
    # VWAP 从第一根K线就有定义，不受窗口限制
    assert result.sub_scores["price_vs_vwap"].score is not None
    # 依然应该正常跑完，不报错
    assert 0.0 <= result.confidence_score <= 100.0


def test_intraday_pipeline_earnings_event_lowers_confidence():
    ohlcv = make_intraday_ohlcv(n=60)

    baseline = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv, rsi=45.0)
    with_earnings = run_intraday_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        ohlcv=ohlcv,
        rsi=45.0,
        earnings_date=date(2026, 8, 7),  # 2 个交易日后，落在默认 3 天窗口内
    )

    assert with_earnings.raw_confidence_score == baseline.raw_confidence_score  # Bull/Bear/原始置信度不受事件影响
    assert with_earnings.confidence_score == baseline.raw_confidence_score * 0.5
    assert "财报临近，历史规律可能失效" in with_earnings.labels
    assert len(with_earnings.event_adjustments) == 1


def test_intraday_pipeline_score_kwargs_override_weights():
    ohlcv = make_intraday_ohlcv(n=60)
    default_result = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv, rsi=45.0)
    overridden_result = run_intraday_pipeline(
        symbol="AAPL",
        current_date=date(2026, 8, 5),
        ohlcv=ohlcv,
        rsi=45.0,
        score_kwargs={"weights": {"rsi": 0.9}},
    )
    assert overridden_result.sub_scores["rsi"].weight == 0.9
    assert default_result.sub_scores["rsi"].weight != 0.9
