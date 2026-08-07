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


def test_intraday_pipeline_rsi_real_value_flows_through_correctly():
    # 验证 rsi 不只是"传了就不缺失"，而是确实按 score_intraday 的公式正确
    # 参与了计算——用两个不同的 RSI 值（其它输入完全不变）核对：
    # 1. sub_scores["rsi"].raw_value 精确等于传入的 rsi；
    # 2. sub_scores["rsi"].score 精确等于 score_intraday 内部公式算出的值
    #    （score = linear_map(rsi,0,100,100,-100) = 100 - 2*rsi）；
    # 3. 换一个 RSI 值，最终 Bull/Bear Score 按 RSI 权重(0.20)精确变化——
    #    证明这个值真的传进了打分逻辑、参与了 Bull/Bear 汇总，不是摆在
    #    sub_scores 里但没被用上。
    ohlcv = make_intraday_ohlcv(n=60)

    oversold = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv, rsi=20.0)
    overbought = run_intraday_pipeline(symbol="AAPL", current_date=date(2026, 8, 5), ohlcv=ohlcv, rsi=80.0)

    # raw_value 精确传递
    assert oversold.sub_scores["rsi"].raw_value == 20.0
    assert overbought.sub_scores["rsi"].raw_value == 80.0

    # score 精确等于 linear_map 公式的结果：rsi=20 -> 100-40=60（超卖看多）；
    # rsi=80 -> 100-160=-60（超买看空）
    assert oversold.sub_scores["rsi"].score == 60.0
    assert overbought.sub_scores["rsi"].score == -60.0

    # 两次调用其它输入完全相同，Bull/Bear Score 的差异应该精确等于
    # RSI 权重(0.20，DEFAULT_WEIGHTS["rsi"])乘以两次 RSI 子分数的差异
    # （60 vs -60，各自只贡献到 Bull 或 Bear 的对应一侧）：
    #   bull(oversold) - bull(overbought) = 0.20 * (60 - 0) = 12.0
    #   bear(overbought) - bear(oversold) = 0.20 * (60 - 0) = 12.0
    assert abs((oversold.bull_score - overbought.bull_score) - 12.0) < 1e-9
    assert abs((overbought.bear_score - oversold.bear_score) - 12.0) < 1e-9


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
