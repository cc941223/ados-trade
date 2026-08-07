"""apply_event_adjustments() / combine_multipliers() 单元测试。

统一用 current_date = 2026-08-05（周三）作为"今天"，交易日窗口手算对照
（用 numpy.busday_count 的语义：只排除周末，不考虑交易所节假日）：

    +1 交易日 = 2026-08-06（周四）
    +2 交易日 = 2026-08-07（周五）
    +3 交易日 = 2026-08-10（周一）
    +4 交易日 = 2026-08-11（周二）
    +5 交易日 = 2026-08-12（周三）
    +6 交易日 = 2026-08-13（周四）
"""
from datetime import date

import pytest

from engine.events.adjustments import apply_event_adjustments, combine_multipliers

TODAY = date(2026, 8, 5)
PLUS = {n: date(2026, 8, 5) for n in [0]}
PLUS.update(
    {
        1: date(2026, 8, 6),
        2: date(2026, 8, 7),
        3: date(2026, 8, 10),
        4: date(2026, 8, 11),
        5: date(2026, 8, 12),
        6: date(2026, 8, 13),
    }
)
MINUS = {
    1: date(2026, 8, 4),  # 周二，恰好是 TODAY 的前 1 个交易日
    2: date(2026, 8, 3),  # 周一，TODAY 的前 2 个交易日
}


# ---------------------------------------------------------------------------
# 1. 财报日历
# ---------------------------------------------------------------------------


def test_earnings_within_window_triggers_confidence_multiplier():
    # 财报在 +2 交易日（窗口内，默认窗口3天）
    result = apply_event_adjustments(scenario="swing", current_date=TODAY, earnings_date=PLUS[2])

    assert result.confidence_multiplier == 0.5
    assert "财报临近，历史规律可能失效" in result.labels
    assert any(a["event_type"] == "earnings" and a["adjustment_type"] == "confidence_multiplier" for a in result.adjustments)


def test_earnings_outside_window_no_effect():
    # 财报在 +5 交易日（超出默认窗口3天）
    result = apply_event_adjustments(scenario="swing", current_date=TODAY, earnings_date=PLUS[5])

    assert result.confidence_multiplier == 1.0
    assert result.labels == []


def test_earnings_unconfirmed_label_fires_outside_proximity_window():
    # 财报在 +5 交易日（超出临近窗口），但日期未确认——按当前实现，这条标注
    # 不要求落在3天窗口内（见 adjustments.py 里的说明与本次对话的确认）。
    result = apply_event_adjustments(
        scenario="swing", current_date=TODAY, earnings_date=PLUS[5], earnings_confirmed=False
    )

    assert result.confidence_multiplier == 1.0  # 只是标签，不影响置信度
    assert "日期未确认，谨慎安排到期日" in result.labels
    assert "财报临近，历史规律可能失效" not in result.labels


def test_earnings_unconfirmed_label_coexists_with_proximity_trigger():
    result = apply_event_adjustments(
        scenario="swing", current_date=TODAY, earnings_date=PLUS[2], earnings_confirmed=False
    )

    assert result.confidence_multiplier == 0.5
    assert "财报临近，历史规律可能失效" in result.labels
    assert "日期未确认，谨慎安排到期日" in result.labels


def test_earnings_not_applicable_to_leveraged_etf_scenario():
    result = apply_event_adjustments(scenario="leveraged_etf", current_date=TODAY, earnings_date=PLUS[2])

    assert result.confidence_multiplier == 1.0
    assert result.labels == []
    assert result.adjustments == []


def test_earnings_already_happened_no_effect():
    # 财报日期已经过去（PLUS[0] 即今天本身，days_until==0，不满足 "> 0"）
    result = apply_event_adjustments(scenario="swing", current_date=TODAY, earnings_date=PLUS[0])
    assert result.confidence_multiplier == 1.0
    assert result.labels == []


# ---------------------------------------------------------------------------
# 2. OpEx / Triple Witching
# ---------------------------------------------------------------------------


def test_opex_within_window_shifts_max_pain_weight():
    # OpEx 在 +2 交易日（默认窗口2天，含边界）
    result = apply_event_adjustments(scenario="options", current_date=TODAY, opex_date=PLUS[2])

    assert result.max_pain_weight == 0.35
    assert any(a["event_type"] == "opex" and a["adjustment_type"] == "weight_shift" for a in result.adjustments)


def test_opex_on_the_day_itself_included_in_window():
    # OpEx 当天（days_until==0）也应触发权重上调
    result = apply_event_adjustments(scenario="options", current_date=TODAY, opex_date=PLUS[0])
    assert result.max_pain_weight == 0.35


def test_opex_outside_window_no_shift():
    # +3 交易日，超出默认窗口2天
    result = apply_event_adjustments(scenario="options", current_date=TODAY, opex_date=PLUS[3])
    assert result.max_pain_weight is None


def test_triple_witching_label_only_on_opex_day():
    result = apply_event_adjustments(
        scenario="options", current_date=TODAY, opex_date=PLUS[0], is_triple_witching=True
    )
    assert "三重魔力日，Gamma pinning 效应可能加剧" in result.labels
    assert result.max_pain_weight == 0.35  # 当天同时也在权重上调窗口内


def test_triple_witching_flag_without_opex_today_has_no_effect():
    # is_triple_witching=True 但 OpEx 还没到（不是今天），不应该出现标签
    result = apply_event_adjustments(
        scenario="options", current_date=TODAY, opex_date=PLUS[2], is_triple_witching=True
    )
    assert "三重魔力日，Gamma pinning 效应可能加剧" not in result.labels


def test_opex_not_applicable_outside_options_scenario():
    result = apply_event_adjustments(scenario="swing", current_date=TODAY, opex_date=PLUS[0])
    assert result.max_pain_weight is None


# ---------------------------------------------------------------------------
# 3. 除权除息日
# ---------------------------------------------------------------------------


def test_ex_dividend_warns_with_deep_itm_covered_call():
    result = apply_event_adjustments(
        scenario="options",
        current_date=TODAY,
        ex_dividend_date=PLUS[3],  # 默认窗口3天，边界值
        holds_deep_itm_covered_call=True,
    )
    assert "提前指派风险" in result.independent_warnings
    # 独立警示不影响 confidence_multiplier
    assert result.confidence_multiplier == 1.0


def test_ex_dividend_no_warning_without_covered_call():
    result = apply_event_adjustments(
        scenario="options", current_date=TODAY, ex_dividend_date=PLUS[2], holds_deep_itm_covered_call=False
    )
    assert result.independent_warnings == []


def test_ex_dividend_outside_window_no_warning():
    result = apply_event_adjustments(
        scenario="options", current_date=TODAY, ex_dividend_date=PLUS[4], holds_deep_itm_covered_call=True
    )
    assert result.independent_warnings == []


def test_ex_dividend_not_applicable_outside_options_scenario():
    result = apply_event_adjustments(
        scenario="swing", current_date=TODAY, ex_dividend_date=PLUS[2], holds_deep_itm_covered_call=True
    )
    assert result.independent_warnings == []


# ---------------------------------------------------------------------------
# 4. 指数再平衡
# ---------------------------------------------------------------------------


def test_rebalance_within_window_for_leveraged_etf():
    result = apply_event_adjustments(scenario="leveraged_etf", current_date=TODAY, rebalance_date=PLUS[5])
    assert result.confidence_multiplier == 0.7


def test_rebalance_outside_window_no_effect():
    result = apply_event_adjustments(scenario="leveraged_etf", current_date=TODAY, rebalance_date=PLUS[6])
    assert result.confidence_multiplier == 1.0


def test_rebalance_not_applicable_outside_leveraged_etf_scenario():
    result = apply_event_adjustments(scenario="options", current_date=TODAY, rebalance_date=PLUS[5])
    assert result.confidence_multiplier == 1.0


# ---------------------------------------------------------------------------
# 5. 宏观事件（所有场景生效）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["intraday", "swing", "options", "leveraged_etf"])
def test_macro_same_day_applies_to_all_scenarios(scenario):
    result = apply_event_adjustments(scenario=scenario, current_date=TODAY, macro_event_date=TODAY)
    assert result.confidence_multiplier == 0.6


def test_macro_next_day():
    # 宏观事件发生在前1个交易日
    result = apply_event_adjustments(scenario="intraday", current_date=TODAY, macro_event_date=MINUS[1])
    assert result.confidence_multiplier == 0.85


def test_macro_second_day_no_effect():
    # 宏观事件发生在前2个交易日，第2个交易日起恢复正常
    result = apply_event_adjustments(scenario="intraday", current_date=TODAY, macro_event_date=MINUS[2])
    assert result.confidence_multiplier == 1.0


def test_macro_event_in_future_no_effect():
    # 宏观事件规则只定义了"当天"和"后1个交易日"，事件还没发生时不生效
    result = apply_event_adjustments(scenario="intraday", current_date=TODAY, macro_event_date=PLUS[1])
    assert result.confidence_multiplier == 1.0


# ---------------------------------------------------------------------------
# 6. 多事件叠加（combine_multipliers 方案）
# ---------------------------------------------------------------------------


def test_combine_multipliers_empty_is_one():
    assert combine_multipliers([]) == 1.0


def test_combine_multipliers_multiplies_all():
    assert abs(combine_multipliers([0.5, 0.6, 0.7]) - 0.21) < 1e-9


def test_earnings_and_macro_same_day_stack_multiplicatively():
    # 财报窗口内(+2交易日, ×0.5) 同时叠加宏观事件当天(×0.6)
    # 预期 confidence_multiplier = 0.5 * 0.6 = 0.3
    result = apply_event_adjustments(
        scenario="swing", current_date=TODAY, earnings_date=PLUS[2], macro_event_date=TODAY
    )
    assert abs(result.confidence_multiplier - 0.3) < 1e-9
    assert len(result.adjustments) == 2


def test_leveraged_etf_scenario_only_macro_applies_when_earnings_also_present():
    # earnings_date 提供了，但 leveraged_etf 场景不适用财报规则，
    # 只有宏观事件生效 -> multiplier 应该是单独的 0.6，不是 0.5*0.6
    result = apply_event_adjustments(
        scenario="leveraged_etf", current_date=TODAY, earnings_date=PLUS[2], macro_event_date=TODAY
    )
    assert result.confidence_multiplier == 0.6


def test_rebalance_and_macro_stack_for_leveraged_etf():
    # 再平衡窗口(×0.7) + 宏观事件当天(×0.6) 同时触发
    result = apply_event_adjustments(
        scenario="leveraged_etf", current_date=TODAY, rebalance_date=PLUS[5], macro_event_date=TODAY
    )
    assert abs(result.confidence_multiplier - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# 7. 其它边界情况
# ---------------------------------------------------------------------------


def test_invalid_scenario_raises():
    with pytest.raises(ValueError):
        apply_event_adjustments(scenario="not_a_real_scenario", current_date=TODAY)


def test_no_events_returns_neutral_result():
    result = apply_event_adjustments(scenario="swing", current_date=TODAY)
    assert result.confidence_multiplier == 1.0
    assert result.max_pain_weight is None
    assert result.labels == []
    assert result.independent_warnings == []
    assert result.adjustments == []


def test_symbol_and_scenario_propagate_into_adjustments_audit_trail():
    result = apply_event_adjustments(
        scenario="options", current_date=TODAY, symbol="TSLA", opex_date=PLUS[0]
    )
    assert result.adjustments[0]["symbol"] == "TSLA"
    assert result.adjustments[0]["scenario"] == "options"
    assert result.adjustments[0]["ts"] == TODAY


def test_custom_coefficients_override_defaults():
    # 权重表/系数支持外部传参覆盖（跟 engine.scoring 的 weights 参数同理）
    result = apply_event_adjustments(
        scenario="swing",
        current_date=TODAY,
        earnings_date=PLUS[2],
        earnings_confidence_multiplier=0.8,
        earnings_window_days=5,
    )
    assert result.confidence_multiplier == 0.8
