"""score_options() 单元测试 —— 手算对照。

场景 A（全部子指标同向看多，边界值饱和）：
    iv_skew=-0.05（=-iv_skew_scale，负skew，看多）           -> +100
    iv_term_structure_ratio=0.85（raw=-0.15=-scale，contango）-> +100
    max_pain=103, price=100（raw=+0.03=scale，Max Pain在上方）-> +100
    iv_percentile=80（不影响方向，强制记 0 分）                -> 0
    earnings_surprise_avg=0.05（=scale，财报窗口期内）          -> +100

    权重 0.25/0.20/0.20/0.20/0.15：
    Bull = .25*100+.20*100+.20*100+.20*0+.15*100 = 25+20+20+0+15 = 80
    Bear = 0
    completeness=1
    strength=(80+0)/100=0.8, agreement=|80-0|/80=1.0, conviction=0.8
    confidence = (1*0.5 + 0.8*0.5)*100 = 90
    （Bull 没有饱和到 100——因为 iv_percentile 子指标恒记 0 分，稀释了整体强度，
    所以即便完全同向也不是满分置信度）

场景 B（子指标互相矛盾）：
    iv_skew=+0.05（正skew，看空）        -> -100
    iv_term_structure_ratio=1.15（backwardation）-> -100
    max_pain/earnings 同场景 A -> +100（看多）
    iv_percentile 任意值 -> 0

    Bull = .20*100(max_pain) + .15*100(earnings) = 20+15 = 35
    Bear = .25*100(skew) + .20*100(term) = 25+20 = 45
    completeness=1
    strength=(35+45)/100=0.8, agreement=|35-45|/80=0.125, conviction=0.8*0.125=0.1
    confidence=(1*0.5+0.1*0.5)*100=55
"""
import pytest

from engine.scoring.options import score_options


def test_score_options_all_bullish_manual():
    result = score_options(
        price=100,
        iv_skew=-0.05,
        iv_term_structure_ratio=0.85,
        max_pain=103,
        iv_percentile=80,
        earnings_surprise_avg=0.05,
        in_earnings_window=True,
    )

    assert abs(result.bull_score - 80.0) < 1e-6
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 90.0) < 1e-6


def test_score_options_contradictory_manual():
    result = score_options(
        price=100,
        iv_skew=0.05,
        iv_term_structure_ratio=1.15,
        max_pain=103,
        iv_percentile=20,
        earnings_surprise_avg=0.05,
        in_earnings_window=True,
    )

    assert abs(result.bull_score - 35.0) < 1e-6
    assert abs(result.bear_score - 45.0) < 1e-6
    assert abs(result.confidence_score - 55.0) < 1e-6


def test_score_options_contradictory_confidence_lower_than_aligned():
    aligned = score_options(
        price=100, iv_skew=-0.05, iv_term_structure_ratio=0.85, max_pain=103,
        iv_percentile=80, earnings_surprise_avg=0.05, in_earnings_window=True,
    )
    contradictory = score_options(
        price=100, iv_skew=0.05, iv_term_structure_ratio=1.15, max_pain=103,
        iv_percentile=20, earnings_surprise_avg=0.05, in_earnings_window=True,
    )
    assert aligned.confidence_score - contradictory.confidence_score > 30


def test_score_options_iv_percentile_never_affects_direction():
    # IV Percentile 规格书明确"影响策略选择而非方向"，无论取值多少都不应
    # 改变 Bull/Bear Score，只应该被记录（供留痕/完整度使用）。
    low = score_options(price=100, iv_skew=-0.05, iv_percentile=5)
    high = score_options(price=100, iv_skew=-0.05, iv_percentile=95)

    assert low.bull_score == high.bull_score
    assert low.bear_score == high.bear_score
    assert low.sub_scores["iv_percentile"].score == 0.0
    assert high.sub_scores["iv_percentile"].score == 0.0


def test_score_options_earnings_surprise_excluded_outside_window():
    # 不在财报窗口期时，历史财报意外幅度子指标视为不适用（等同缺失）
    result = score_options(
        price=100,
        iv_skew=-0.05,
        iv_term_structure_ratio=0.85,
        max_pain=103,
        iv_percentile=80,
        earnings_surprise_avg=0.05,
        in_earnings_window=False,
    )
    assert result.sub_scores["earnings_surprise"].score is None
    # 完整度 = (1 - earnings权重0.15) / 1.0 = 0.85
    assert abs(result.extra["completeness"] - 0.85) < 1e-9


def test_score_options_unknown_weight_key_raises():
    with pytest.raises(ValueError):
        score_options(price=100, weights={"not_real": 0.1})
