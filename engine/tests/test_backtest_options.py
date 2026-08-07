"""evaluate_options_signal() 单元测试 —— 手算对照。

全程只用 bid/ask，不引入 mid price 列，验证 6.3 节"买一/卖一价模拟真实
成交"这条硬性要求确实被遵守。
"""
import pandas as pd
import pytest

from engine.backtest.options import evaluate_options_signal


def test_long_profit_target_hit_uses_bid():
    # entry=2.00（买入付出的权利金），多头平仓吃 bid
    # day0: bid=2.50 -> pnl=(2.50-2.00)/2.00=0.25，未达50%，dte=40未到21
    # day1: bid=3.10 -> pnl=(3.10-2.00)/2.00=0.55 >=0.5，触发目标平仓
    price_path = pd.DataFrame(
        {"bid": [2.50, 3.10], "ask": [2.60, 3.20], "dte": [40, 39]}
    )

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")

    assert result.outcome == "win"
    assert result.exit_reason == "target_hit"
    assert result.exit_price == 3.10  # 用的是 bid，不是 ask 或 mid
    assert abs(result.raw_return_pct - 0.55) < 1e-9


def test_long_time_exit_with_profit_is_win():
    # entry=2.00，一直没达到50%盈利，dte 降到21时强制平仓
    # day0: bid=2.10, dte=22 -> pnl=0.05，未到21天
    # day1: bid=2.20, dte=21 -> pnl=0.10，dte<=21 触发到期强制平仓，pnl>=0 -> win
    price_path = pd.DataFrame({"bid": [2.10, 2.20], "ask": [2.15, 2.25], "dte": [22, 21]})

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")

    assert result.outcome == "win"
    assert result.exit_reason == "time_exit"
    assert result.exit_price == 2.20
    assert abs(result.raw_return_pct - 0.10) < 1e-9


def test_long_time_exit_with_loss_is_loss():
    # entry=2.00, day0: bid=1.50, dte=21 -> pnl=(1.50-2.00)/2.00=-0.25, 触发到期平仓, pnl<0 -> loss
    price_path = pd.DataFrame({"bid": [1.50], "ask": [1.55], "dte": [21]})

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")

    assert result.outcome == "loss"
    assert result.exit_reason == "time_exit"
    assert abs(result.raw_return_pct - (-0.25)) < 1e-9


def test_long_max_drawdown_tracked_independently_of_final_outcome():
    # entry=2.00
    # day0: bid=1.00 (pnl=-0.5，最大浮亏)  dte=40
    # day1: bid=1.20 (pnl=-0.4)            dte=39
    # day2: bid=3.10 (pnl=0.55, >=0.5)     dte=38 -> 触发目标平仓, win
    # 最终 win，但持仓期间经历过 -0.5 的最大浮亏，两者要分开记录
    price_path = pd.DataFrame(
        {"bid": [1.00, 1.20, 3.10], "ask": [1.05, 1.25, 3.20], "dte": [40, 39, 38]}
    )

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")

    assert result.outcome == "win"
    assert abs(result.raw_return_pct - 0.55) < 1e-9
    assert abs(result.max_drawdown_pct - (-0.5)) < 1e-9


def test_short_position_closes_at_ask_not_bid():
    # entry=2.00（卖出收到的权利金），空头平仓（买回）吃 ask
    # day0: ask=1.00 -> pnl=(2.00-1.00)/2.00=0.50 >=0.5 触发目标平仓
    price_path = pd.DataFrame({"bid": [0.90], "ask": [1.00], "dte": [40]})

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="short")

    assert result.outcome == "win"
    assert result.exit_price == 1.00  # 用的是 ask，不是 bid
    assert abs(result.raw_return_pct - 0.50) < 1e-9


def test_target_and_time_exit_same_day_prefers_target():
    # day0: bid=3.10 (pnl=0.55>=0.5), dte=21（同时满足两条件）-> 按目标平仓处理
    price_path = pd.DataFrame({"bid": [3.10], "ask": [3.20], "dte": [21]})

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")
    assert result.exit_reason == "target_hit"


def test_undecided_when_data_ends_before_any_exit_condition():
    # 一直没到50%盈利，dte也一直没降到21
    price_path = pd.DataFrame({"bid": [2.10, 2.20], "ask": [2.15, 2.25], "dte": [40, 35]})

    result = evaluate_options_signal(entry_price=2.00, price_path=price_path, position_side="long")

    assert result.outcome == "undecided"
    assert result.exit_reason == "undecided"
    assert result.exit_price == 2.20  # 仍记录最后一次观测值，方便留痕
    assert result.max_drawdown_pct is not None


def test_custom_profit_target_and_time_exit_dte():
    # profit_target_pct=0.3, time_exit_dte=10
    # day0: bid=2.60 -> pnl=0.30 >=0.3 触发
    price_path = pd.DataFrame({"bid": [2.60], "ask": [2.65], "dte": [30]})

    result = evaluate_options_signal(
        entry_price=2.00, price_path=price_path, position_side="long", profit_target_pct=0.3, time_exit_dte=10
    )
    assert result.outcome == "win"
    assert result.exit_reason == "target_hit"


def test_invalid_position_side_raises():
    price_path = pd.DataFrame({"bid": [2.0], "ask": [2.1], "dte": [30]})
    with pytest.raises(ValueError):
        evaluate_options_signal(entry_price=2.0, price_path=price_path, position_side="both")
