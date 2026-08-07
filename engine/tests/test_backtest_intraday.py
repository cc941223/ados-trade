"""evaluate_intraday_signal() 单元测试 —— 手算对照。"""
import pandas as pd
import pytest

from engine.backtest.intraday import evaluate_intraday_signal


def test_target_hit_first_bull():
    # entry=100, atr=2, bull方向: target=100+1.5*2=103, stop=100-1*2=98
    # bar1: high=101,low=99 -> 都没触及
    # bar2: high=104,low=100 -> high>=103 触及目标，且 low(100)>98 没触及止损
    price_path = pd.DataFrame({"high": [101, 104], "low": [99, 100]})

    result = evaluate_intraday_signal(entry_price=100, direction="bull", atr=2, price_path=price_path)

    assert result.outcome == "win"
    assert result.exit_reason == "target_hit"
    assert result.exit_price == 103
    assert result.exit_ts == 1
    assert abs(result.raw_return_pct - 0.03) < 1e-9  # (103-100)/100


def test_stop_hit_first_bull():
    # entry=100, atr=2, target=103, stop=98
    # bar1: high=101,low=97 -> low<=98 触及止损，high(101)<103 没触及目标
    price_path = pd.DataFrame({"high": [101], "low": [97]})

    result = evaluate_intraday_signal(entry_price=100, direction="bull", atr=2, price_path=price_path)

    assert result.outcome == "loss"
    assert result.exit_reason == "stop_hit"
    assert result.exit_price == 98
    assert abs(result.raw_return_pct - (-0.02)) < 1e-9  # (98-100)/100


def test_neither_hit_is_undecided():
    # target=103, stop=98，K线始终在[98,103]区间内部
    price_path = pd.DataFrame(
        {"high": [101, 102, 100.5], "low": [99, 99.5, 99], "close": [100.5, 101, 100]}
    )

    result = evaluate_intraday_signal(entry_price=100, direction="bull", atr=2, price_path=price_path)

    assert result.outcome == "undecided"
    assert result.exit_reason == "undecided"
    assert result.exit_price == 100  # 最后一根收盘价
    assert abs(result.raw_return_pct - 0.0) < 1e-9


def test_bear_direction_target_and_stop_mirrored():
    # entry=100, atr=2, bear方向: target=100-1.5*2=97, stop=100+1*2=102
    # bar1: high=101,low=96 -> low<=97 触及目标, high(101)<102 没触及止损
    price_path = pd.DataFrame({"high": [101], "low": [96]})

    result = evaluate_intraday_signal(entry_price=100, direction="bear", atr=2, price_path=price_path)

    assert result.outcome == "win"
    assert result.exit_reason == "target_hit"
    assert result.exit_price == 97
    # bear方向下跌是正收益: (100-97)/100=0.03
    assert abs(result.raw_return_pct - 0.03) < 1e-9


def test_bear_direction_stop_hit():
    # entry=100, atr=2, bear: target=97, stop=102
    # bar1: high=103,low=99 -> high>=102 触及止损, low(99)>97 没触及目标
    price_path = pd.DataFrame({"high": [103], "low": [99]})

    result = evaluate_intraday_signal(entry_price=100, direction="bear", atr=2, price_path=price_path)

    assert result.outcome == "loss"
    assert result.exit_reason == "stop_hit"
    assert result.exit_price == 102
    # bear方向下，价格上涨到102是负收益: (100-102)/100=-0.02
    assert abs(result.raw_return_pct - (-0.02)) < 1e-9


def test_same_bar_tie_break_defaults_to_stop_first():
    # entry=100, atr=2, bull: target=103, stop=98
    # 单根K线 high=105,low=95 -> 两个条件都满足，默认按止损优先处理
    price_path = pd.DataFrame({"high": [105], "low": [95]})

    result = evaluate_intraday_signal(entry_price=100, direction="bull", atr=2, price_path=price_path)

    assert result.outcome == "loss"
    assert result.exit_reason == "stop_hit"


def test_same_bar_tie_break_can_switch_to_target_first():
    price_path = pd.DataFrame({"high": [105], "low": [95]})

    result = evaluate_intraday_signal(
        entry_price=100, direction="bull", atr=2, price_path=price_path, same_bar_tie_break="target_first"
    )

    assert result.outcome == "win"
    assert result.exit_reason == "target_hit"


def test_custom_multipliers():
    # N=2, M=0.5: target=100+2*2=104, stop=100-0.5*2=99
    price_path = pd.DataFrame({"high": [104.5], "low": [99.5]})

    result = evaluate_intraday_signal(
        entry_price=100,
        direction="bull",
        atr=2,
        price_path=price_path,
        target_multiplier=2,
        stop_multiplier=0.5,
    )
    assert result.outcome == "win"
    assert result.exit_price == 104


def test_invalid_direction_raises():
    price_path = pd.DataFrame({"high": [101], "low": [99]})
    with pytest.raises(ValueError):
        evaluate_intraday_signal(entry_price=100, direction="up", atr=2, price_path=price_path)


def test_empty_price_path_is_undecided_with_no_exit_price():
    price_path = pd.DataFrame({"high": [], "low": [], "close": []})
    result = evaluate_intraday_signal(entry_price=100, direction="bull", atr=2, price_path=price_path)
    assert result.outcome == "undecided"
    assert result.exit_price is None
    assert result.raw_return_pct is None
