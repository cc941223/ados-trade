"""score_intraday() 单元测试 —— 手算对照。

场景 A（全部子指标同向看多）：
    price=101, vwap=100          -> price_vs_vwap_raw=0.01（=vwap_scale，饱和）-> +100
    rsi=20                       -> linear_map(20,0,100,100,-100) = 60
    volume_ratio=2.0, chg=+0.01  -> magnitude=100（=饱和），方向+1 -> +100
    atr=1.0, ref=100, price=101  -> z=1.0，atr_z_scale=2.0 -> linear_map(1,-2,2,-100,100)=50
    short_ma=105, long_ma=100    -> raw=0.05（>ma_scale=0.01，饱和） -> +100

    权重：0.25/0.20/0.25/0.15/0.15
    Bull = .25*100+.20*60+.25*100+.15*50+.15*100 = 25+12+25+7.5+15 = 84.5
    Bear = 0
    completeness=1
    strength=(84.5+0)/100=0.845, agreement=|84.5-0|/84.5=1.0, conviction=0.845
    confidence = (1*0.5 + 0.845*0.5)*100 = 92.25
    （注意：Bull 没有饱和到 100，所以即便完全同向、无矛盾，conviction 也不是
    满分 1——这正是本次修正要解决的问题：证据强度不足时，即使方向一致，也不该
    给出满分置信度。）

场景 B（子指标互相矛盾）：
    price_vs_vwap 同场景 A -> +100（看多）
    rsi=90（超买）          -> linear_map(90,0,100,100,-100) = -80（看空）
    volume_ratio=2.0, chg=-0.01 -> -100（看空）
    atr 同场景 A -> +50（看多）
    short_ma=95,long_ma=100 -> raw=-0.05（饱和） -> -100（看空）

    Bull = .25*100 + .15*50 = 25+7.5 = 32.5
    Bear = .20*80 + .25*100 + .15*100 = 16+25+15 = 56
    completeness=1
    strength=(32.5+56)/100=0.885, agreement=|32.5-56|/88.5=0.265536...
    conviction=0.885*0.265536...=0.235
    confidence = (1*0.5+0.235*0.5)*100 = 61.75
"""
from engine.scoring.intraday import DEFAULT_WEIGHTS, score_intraday


def test_score_intraday_all_bullish_manual():
    result = score_intraday(
        price=101,
        vwap=100,
        rsi=20,
        volume_ratio=2.0,
        price_change_pct=0.01,
        atr=1.0,
        reference_price=100,
        short_ma=105,
        long_ma=100,
    )

    assert abs(result.bull_score - 84.5) < 1e-6
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 92.25) < 1e-6


def test_score_intraday_contradictory_manual():
    result = score_intraday(
        price=101,
        vwap=100,
        rsi=90,
        volume_ratio=2.0,
        price_change_pct=-0.01,
        atr=1.0,
        reference_price=100,
        short_ma=95,
        long_ma=100,
    )

    assert abs(result.bull_score - 32.5) < 1e-6
    assert abs(result.bear_score - 56.0) < 1e-6
    assert abs(result.confidence_score - 61.75) < 1e-3


def test_score_intraday_contradictory_confidence_lower_than_aligned():
    aligned = score_intraday(
        price=101,
        vwap=100,
        rsi=20,
        volume_ratio=2.0,
        price_change_pct=0.01,
        atr=1.0,
        reference_price=100,
        short_ma=105,
        long_ma=100,
    )
    contradictory = score_intraday(
        price=101,
        vwap=100,
        rsi=90,
        volume_ratio=2.0,
        price_change_pct=-0.01,
        atr=1.0,
        reference_price=100,
        short_ma=95,
        long_ma=100,
    )
    assert aligned.confidence_score - contradictory.confidence_score > 30


def test_score_intraday_missing_volume_ratio_lowers_completeness():
    # 没有 price_change_pct 时无法判断放量方向，量比子指标视为缺失
    result = score_intraday(price=101, vwap=100, volume_ratio=2.0, price_change_pct=None)
    assert result.sub_scores["volume_ratio"].score is None
    assert result.extra["completeness"] < 1.0


def test_score_intraday_weight_override():
    default_result = score_intraday(price=101, vwap=100, rsi=20)
    overridden_result = score_intraday(price=101, vwap=100, rsi=20, weights={"price_vs_vwap": 0.9})

    assert default_result.sub_scores["price_vs_vwap"].weight == DEFAULT_WEIGHTS["price_vs_vwap"]
    assert overridden_result.sub_scores["price_vs_vwap"].weight == 0.9
    # 权重表其余项保持默认值不受影响
    assert overridden_result.sub_scores["rsi"].weight == DEFAULT_WEIGHTS["rsi"]


def test_score_intraday_unknown_weight_key_raises():
    import pytest

    with pytest.raises(ValueError):
        score_intraday(price=101, vwap=100, weights={"not_a_real_indicator": 0.5})
