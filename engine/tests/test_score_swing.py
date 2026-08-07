"""score_swing() 单元测试 —— 手算对照。

场景 A（全部子指标同向看多，全部饱和到 ±100）：
    rs_value=20 (>rs_scale=15)                        -> +100
    price=115, ma50=110, ma200=105 (>ma_scale=0.05)   -> +100
    poc=110, price=115 (>poc_scale=0.03)              -> +100
    put_call_ratio=1.5 (raw=+0.5=pcr_scale, 极端高PCR，反向解读为看多) -> +100
    benchmark_return=0.03 (>benchmark_scale=0.02)      -> +100

    权重 0.30/0.25/0.20/0.15/0.10 全部看多：
    Bull = 100, Bear = 0, confidence = 100

场景 B（子指标互相矛盾）：
    rs_value=-20   -> -100（看空）
    ma50/ma200/price 同场景 A -> +100（看多）
    poc=120,price=115 -> raw=-0.04167(饱和) -> -100（看空）
    put_call_ratio=1.5 同场景 A -> +100（看多）
    benchmark_return=-0.03 -> -100（看空）

    Bull = .25*100(ma_trend) + .15*100(pcr) = 25+15 = 40
    Bear = .30*100(rs) + .20*100(volume_profile) + .10*100(sector) = 30+20+10 = 60
    completeness=1
    strength=(40+60)/100=1.0（本场景刚好全部子指标都饱和到±100，Bull+Bear=100，
    strength 达到理论上限，这种情况下 conviction 与旧版 consistency 数值相同，
    等于 agreement=|40-60|/100=0.2）
    confidence=(1*0.5+0.2*0.5)*100=60
"""
from engine.scoring.swing import score_swing


def test_score_swing_all_bullish_manual():
    result = score_swing(
        price=115,
        rs_value=20,
        ma50=110,
        ma200=105,
        poc=110,
        put_call_ratio=1.5,
        benchmark_return=0.03,
    )

    assert abs(result.bull_score - 100.0) < 1e-6
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 100.0) < 1e-6


def test_score_swing_contradictory_manual():
    result = score_swing(
        price=115,
        rs_value=-20,
        ma50=110,
        ma200=105,
        poc=120,
        put_call_ratio=1.5,
        benchmark_return=-0.03,
    )

    assert abs(result.bull_score - 40.0) < 1e-6
    assert abs(result.bear_score - 60.0) < 1e-6
    assert abs(result.confidence_score - 60.0) < 1e-6


def test_score_swing_contradictory_confidence_lower_than_aligned():
    aligned = score_swing(price=115, rs_value=20, ma50=110, ma200=105, poc=110, put_call_ratio=1.5, benchmark_return=0.03)
    contradictory = score_swing(price=115, rs_value=-20, ma50=110, ma200=105, poc=120, put_call_ratio=1.5, benchmark_return=-0.03)

    assert aligned.confidence_score - contradictory.confidence_score > 30


def test_score_swing_only_ma50_provided():
    # ma200 缺失时，中长期均线趋势子指标仍可用 ma50 单独计算（不视为缺失）
    result = score_swing(price=115, ma50=110)
    assert result.sub_scores["ma_trend"].score is not None


def test_score_swing_missing_indicators_lower_completeness():
    result = score_swing(price=115, rs_value=20)
    assert result.extra["completeness"] < 1.0
