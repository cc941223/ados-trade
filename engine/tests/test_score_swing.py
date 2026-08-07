"""score_swing() 单元测试 —— 手算对照。

Volume Profile 位置子指标现在按 Value Area 边界真实判断（而不是 POC 偏离度
近似）：价格突破 value_area_high/low 之外时，用"突破距离 / VA 宽度"归一化，
达到 va_breakout_scale（默认 1.0，即 1 个 VA 宽度）饱和到 ±100；价格落在
区间内时，改用相对 POC（或 VA 中点）的偏离方向给一个幅度更小的分数
（上限 va_in_range_scale，默认 30）。

场景 A（全部子指标同向看多，全部饱和到 ±100）：
    rs_value=20 (>rs_scale=15)                        -> +100
    price=115, ma50=110, ma200=105 (>ma_scale=0.05)   -> +100
    value_area_high=100, value_area_low=95（价格115在VA上方突破）
        raw=(115-100)/(100-95)=3.0（>va_breakout_scale=1.0，饱和） -> +100
    put_call_ratio=1.5 (raw=+0.5=pcr_scale, 极端高PCR，反向解读为看多) -> +100
    benchmark_return=0.03 (>benchmark_scale=0.02)      -> +100

    权重 0.30/0.25/0.20/0.15/0.10 全部看多：
    Bull = 100, Bear = 0, confidence = 100

场景 B（子指标互相矛盾）：
    rs_value=-20   -> -100（看空）
    ma50/ma200/price 同场景 A -> +100（看多）
    value_area_high=130, value_area_low=125（价格115在VA下方突破）
        raw=(115-125)/(130-125)=-2.0（超出-va_breakout_scale=1.0，饱和） -> -100（看空）
    put_call_ratio=1.5 同场景 A -> +100（看多）
    benchmark_return=-0.03 -> -100（看空）

    Bull = .25*100(ma_trend) + .15*100(pcr) = 25+15 = 40
    Bear = .30*100(rs) + .20*100(volume_profile) + .10*100(sector) = 30+20+10 = 60
    completeness=1
    strength=(40+60)/100=1.0（本场景刚好全部子指标都饱和到±100，Bull+Bear=100，
    strength 达到理论上限，这种情况下 conviction 等于 agreement=|40-60|/100=0.2）
    confidence=(1*0.5+0.2*0.5)*100=60
"""
from engine.scoring.swing import score_swing


def test_score_swing_all_bullish_manual():
    result = score_swing(
        price=115,
        rs_value=20,
        ma50=110,
        ma200=105,
        value_area_high=100,
        value_area_low=95,
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
        value_area_high=130,
        value_area_low=125,
        put_call_ratio=1.5,
        benchmark_return=-0.03,
    )

    assert abs(result.bull_score - 40.0) < 1e-6
    assert abs(result.bear_score - 60.0) < 1e-6
    assert abs(result.confidence_score - 60.0) < 1e-6


def test_score_swing_contradictory_confidence_lower_than_aligned():
    aligned = score_swing(
        price=115, rs_value=20, ma50=110, ma200=105, value_area_high=100, value_area_low=95,
        put_call_ratio=1.5, benchmark_return=0.03,
    )
    contradictory = score_swing(
        price=115, rs_value=-20, ma50=110, ma200=105, value_area_high=130, value_area_low=125,
        put_call_ratio=1.5, benchmark_return=-0.03,
    )

    assert aligned.confidence_score - contradictory.confidence_score > 30


def test_score_swing_price_inside_value_area_leans_toward_poc():
    # 价格落在 Value Area 区间内：中性偏向 POC 位置，幅度明显小于突破信号。
    # value_area_high=120, value_area_low=100（宽度20），poc=115，price=110
    # raw = (110-115)/20 = -0.25
    # score = linear_map(-0.25, -0.5, 0.5, -30, 30) = -30 + 0.25*60 = -15
    result = score_swing(price=110, value_area_high=120, value_area_low=100, poc=115)

    assert abs(result.sub_scores["volume_profile_position"].raw_value - (-0.25)) < 1e-9
    assert abs(result.sub_scores["volume_profile_position"].score - (-15.0)) < 1e-9
    # 区间内的分数幅度不应超过 va_in_range_scale（默认30），明显弱于突破信号
    assert abs(result.sub_scores["volume_profile_position"].score) <= 30.0


def test_score_swing_price_inside_value_area_without_poc_uses_midpoint():
    # 不提供 poc 时，区间内退化为用 VA 中点做参照
    # value_area_high=120, value_area_low=100 -> 中点=110，price=110 恰好等于中点 -> 中性 0 分
    result = score_swing(price=110, value_area_high=120, value_area_low=100)
    assert result.sub_scores["volume_profile_position"].score == 0.0


def test_score_swing_missing_value_area_marks_indicator_unavailable():
    result = score_swing(price=115, value_area_high=None, value_area_low=100)
    assert result.sub_scores["volume_profile_position"].score is None


def test_score_swing_only_ma50_provided():
    # ma200 缺失时，中长期均线趋势子指标仍可用 ma50 单独计算（不视为缺失）
    result = score_swing(price=115, ma50=110)
    assert result.sub_scores["ma_trend"].score is not None


def test_score_swing_missing_indicators_lower_completeness():
    result = score_swing(price=115, rs_value=20)
    assert result.extra["completeness"] < 1.0
