"""evaluate_swing_signal() 单元测试 —— 手算对照。"""
import pytest

from engine.backtest.swing import evaluate_swing_signal


def test_bull_direction_beats_benchmark_is_win():
    # 标的 100->112 (+12%)，基准 100->105 (+5%)
    # bull方向: raw_return=0.12, excess=0.12-0.05=0.07 > 0 -> win
    result = evaluate_swing_signal(
        entry_price=100, exit_price=112, benchmark_entry_price=100, benchmark_exit_price=105, direction="bull"
    )
    assert result.outcome == "win"
    assert abs(result.raw_return_pct - 0.12) < 1e-9
    assert abs(result.excess_return_pct - 0.07) < 1e-9


def test_bull_direction_underperforms_benchmark_is_loss():
    # 标的 100->103 (+3%)，基准 100->108 (+8%)
    # excess = 0.03-0.08 = -0.05 < 0 -> loss
    result = evaluate_swing_signal(
        entry_price=100, exit_price=103, benchmark_entry_price=100, benchmark_exit_price=108, direction="bull"
    )
    assert result.outcome == "loss"
    assert abs(result.excess_return_pct - (-0.05)) < 1e-9


def test_bear_direction_price_falls_more_than_benchmark_is_win():
    # 标的 100->90 (-10%)，基准 100->98 (-2%)
    # bear方向自身收益 = -(-0.10) = 0.10（跌得越多赚得越多）
    # 基准同方向收益 = -(-0.02) = 0.02
    # excess = 0.10 - 0.02 = 0.08 > 0 -> win
    result = evaluate_swing_signal(
        entry_price=100, exit_price=90, benchmark_entry_price=100, benchmark_exit_price=98, direction="bear"
    )
    assert result.outcome == "win"
    assert abs(result.raw_return_pct - 0.10) < 1e-9
    assert abs(result.excess_return_pct - 0.08) < 1e-9


def test_bear_direction_price_falls_less_than_benchmark_is_loss():
    # 标的 100->97 (-3%) -> bear自身收益=0.03
    # 基准 100->90 (-10%) -> 基准同方向收益=0.10
    # excess = 0.03-0.10 = -0.07 < 0 -> loss（标的跌得没基准多，做空跑输了）
    result = evaluate_swing_signal(
        entry_price=100, exit_price=97, benchmark_entry_price=100, benchmark_exit_price=90, direction="bear"
    )
    assert result.outcome == "loss"
    assert abs(result.excess_return_pct - (-0.07)) < 1e-9


def test_exactly_zero_excess_is_loss_not_win():
    # 规格书原文"超额收益 > 0 才算成功"，严格大于，等于0不算赢
    result = evaluate_swing_signal(
        entry_price=100, exit_price=105, benchmark_entry_price=100, benchmark_exit_price=105, direction="bull"
    )
    assert abs(result.excess_return_pct - 0.0) < 1e-9
    assert result.outcome == "loss"


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        evaluate_swing_signal(entry_price=100, exit_price=105, benchmark_entry_price=100, benchmark_exit_price=102, direction="up")
