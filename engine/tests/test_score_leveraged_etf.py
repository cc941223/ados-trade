"""score_leveraged_etf() 单元测试 —— 手算对照。

场景 A（全部子指标同向看多 / 低波动 regime，全部饱和或恰好到边界 +100）：
    vix=10（<vix_low=15）                                -> +100（截断）
    vix_term_structure_ratio=0.85（raw=-0.15=-scale）     -> +100
    price=105, ma200=100（raw=+0.05=scale）               -> +100
    daily_decay=0.005（=decay_scale）                     -> +100

    权重 0.30/0.25/0.25/0.20 全部看多：
    Bull=100, Bear=0, confidence=100, regime="bull"（diff=100>10）

场景 B（子指标互相矛盾）：
    vix=30（>vix_high=25）              -> -100（看空）
    vix_term_structure_ratio=1.15（backwardation）-> -100（看空）
    price=105, ma200=100 同场景 A       -> +100（看多）
    daily_decay=0.0（中性，恰好映射到0） -> 0（不贡献方向）

    Bull = .25*100(ma200) = 25
    Bear = .30*100(vix) + .25*100(term) = 30+25 = 55
    completeness=1, consistency=|25-55|/80=0.375, confidence=(1*0.5+0.375*0.5)*100=68.75
    diff = 25-55 = -30 < -10 -> regime="bear"
"""
import pytest

from engine.scoring.leveraged_etf import classify_regime, score_leveraged_etf


def test_score_leveraged_etf_all_bullish_manual():
    result = score_leveraged_etf(
        price=105,
        vix=10,
        vix_term_structure_ratio=0.85,
        ma200=100,
        daily_decay=0.005,
    )

    assert abs(result.bull_score - 100.0) < 1e-6
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 100.0) < 1e-6
    assert result.extra["regime"] == "bull"


def test_score_leveraged_etf_contradictory_manual():
    result = score_leveraged_etf(
        price=105,
        vix=30,
        vix_term_structure_ratio=1.15,
        ma200=100,
        daily_decay=0.0,
    )

    assert abs(result.bull_score - 25.0) < 1e-6
    assert abs(result.bear_score - 55.0) < 1e-6
    assert abs(result.confidence_score - 68.75) < 1e-6
    assert result.extra["regime"] == "bear"


def test_score_leveraged_etf_contradictory_confidence_lower_than_aligned():
    aligned = score_leveraged_etf(price=105, vix=10, vix_term_structure_ratio=0.85, ma200=100, daily_decay=0.005)
    contradictory = score_leveraged_etf(price=105, vix=30, vix_term_structure_ratio=1.15, ma200=100, daily_decay=0.0)

    assert aligned.confidence_score - contradictory.confidence_score > 20


def test_classify_regime_thresholds():
    assert classify_regime(bull_score=50, bear_score=10, threshold=10.0) == "bull"
    assert classify_regime(bull_score=10, bear_score=50, threshold=10.0) == "bear"
    # 恰好等于阈值时判为 chop（严格大于/小于才算方向明确）
    assert classify_regime(bull_score=20, bear_score=10, threshold=10.0) == "chop"
    assert classify_regime(bull_score=50, bear_score=50, threshold=10.0) == "chop"


def test_score_leveraged_etf_missing_indicator_lowers_completeness():
    result = score_leveraged_etf(vix=10)
    assert result.extra["completeness"] < 1.0


def test_score_leveraged_etf_unknown_weight_key_raises():
    with pytest.raises(ValueError):
        score_leveraged_etf(vix=10, weights={"not_real": 0.1})
