"""score_leveraged_etf() 单元测试 —— 手算对照。

场景 A（全部子指标同向看多 / 低波动 regime，全部饱和或恰好到边界 +100）：
    vix=10（<vix_low=15）                                -> +100（截断）
    vix_term_structure_ratio=0.85（raw=-0.15=-scale）     -> +100
    underlying_price=105, underlying_ma200=100（raw=+0.05=scale） -> +100
    daily_decay=0.005（=decay_scale）                     -> +100

    权重 0.30/0.25/0.25/0.20 全部看多：
    Bull=100, Bear=0, confidence=100, regime="bull"（diff=100>10）

场景 B（子指标互相矛盾）：
    vix=30（>vix_high=25）              -> -100（看空）
    vix_term_structure_ratio=1.15（backwardation）-> -100（看空）
    underlying_price=105, underlying_ma200=100 同场景 A -> +100（看多）
    daily_decay=0.0（中性，恰好映射到0） -> 0（不贡献方向）

    Bull = .25*100(ma200) = 25
    Bear = .30*100(vix) + .25*100(term) = 30+25 = 55
    completeness=1
    strength=(25+55)/100=0.8, agreement=|25-55|/80=0.375, conviction=0.8*0.375=0.3
    confidence=(1*0.5+0.3*0.5)*100=65
    diff = 25-55 = -30 < -10 -> regime="bear"

以上两个场景默认 target_type="broad_index"（不传等价于用默认值），单独用
`test_target_type_*` 系列验证 target_type="single_stock" 时确实换了输入源
（200日均线位置、波动损耗率两个子指标），且 VIX 两个子指标的处理逻辑不变。
"""
import pytest

from engine.scoring.leveraged_etf import VALID_TARGET_TYPES, classify_regime, score_leveraged_etf


def test_score_leveraged_etf_all_bullish_manual():
    result = score_leveraged_etf(
        underlying_price=105,
        vix=10,
        vix_term_structure_ratio=0.85,
        underlying_ma200=100,
        daily_decay=0.005,
    )

    assert abs(result.bull_score - 100.0) < 1e-6
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 100.0) < 1e-6
    assert result.extra["regime"] == "bull"


def test_score_leveraged_etf_contradictory_manual():
    result = score_leveraged_etf(
        underlying_price=105,
        vix=30,
        vix_term_structure_ratio=1.15,
        underlying_ma200=100,
        daily_decay=0.0,
    )

    assert abs(result.bull_score - 25.0) < 1e-6
    assert abs(result.bear_score - 55.0) < 1e-6
    assert abs(result.confidence_score - 65.0) < 1e-6
    assert result.extra["regime"] == "bear"


def test_score_leveraged_etf_contradictory_confidence_lower_than_aligned():
    aligned = score_leveraged_etf(
        underlying_price=105, vix=10, vix_term_structure_ratio=0.85, underlying_ma200=100, daily_decay=0.005
    )
    contradictory = score_leveraged_etf(
        underlying_price=105, vix=30, vix_term_structure_ratio=1.15, underlying_ma200=100, daily_decay=0.0
    )

    assert aligned.confidence_score - contradictory.confidence_score > 20


# ---------------------------------------------------------------------------
# target_type 区分：broad_index vs single_stock
# ---------------------------------------------------------------------------


def test_target_type_defaults_to_broad_index():
    # 不传 target_type 时默认 broad_index，跟显式传 broad_index 结果一致
    default_result = score_leveraged_etf(underlying_price=105, underlying_ma200=100, vix=10)
    explicit_result = score_leveraged_etf(
        target_type="broad_index", underlying_price=105, underlying_ma200=100, vix=10
    )
    assert default_result.bull_score == explicit_result.bull_score
    assert default_result.bear_score == explicit_result.bear_score


def test_target_type_single_stock_uses_the_stock_own_ma200_not_broad_market():
    # 同样的数字，分别当作"QQQ 自己的价格/均线"（broad_index）和
    # "TSLA 自己的价格/均线"（single_stock）传入——计算逻辑完全相同
    # （公式不因 target_type 而变），只是数据含义不同，两次调用结果应一致，
    # 说明函数本身不会因为换了 target_type 就改变 200 日均线位置的算法，
    # 差异完全来自调用方传入的数值本身。
    qqq_price, qqq_ma200 = 480.0, 460.0  # 假设的 QQQ 自身价格与200日均线
    tsla_price, tsla_ma200 = 250.0, 240.0  # 假设的 TSLA 自身价格与200日均线

    broad_index_result = score_leveraged_etf(
        target_type="broad_index", underlying_price=qqq_price, underlying_ma200=qqq_ma200, vix=18
    )
    single_stock_result = score_leveraged_etf(
        target_type="single_stock", underlying_price=tsla_price, underlying_ma200=tsla_ma200, vix=18
    )

    # QQQ: raw=(480-460)/460=0.043478...; TSLA: raw=(250-240)/240=0.041666...
    # 两者接近但不同，且都在 ma_scale=0.05 以内、未饱和，可用于验证两次调用
    # 确实各自独立使用了传入的（不同标的的）underlying_price/underlying_ma200，
    # 而不是共享或混用了同一组"大盘"数据。
    assert (
        abs(broad_index_result.sub_scores["ma200_position"].raw_value - (480 - 460) / 460) < 1e-9
    )
    assert (
        abs(single_stock_result.sub_scores["ma200_position"].raw_value - (250 - 240) / 240) < 1e-9
    )
    assert broad_index_result.sub_scores["ma200_position"].raw_value != single_stock_result.sub_scores[
        "ma200_position"
    ].raw_value


def test_target_type_single_stock_higher_volatility_decay_is_not_an_error():
    # single_stock 的波动损耗率数值通常显著更负（损耗更剧烈），公式和饱和
    # 阈值不变，只是数值本身更极端——用同样的 decay_scale 验证两者都能正常
    # 饱和到满分看空，不会因为数值"看起来太大"而报错或被特殊处理。
    broad_index_result = score_leveraged_etf(target_type="broad_index", vix=18, daily_decay=-0.001)
    single_stock_result = score_leveraged_etf(target_type="single_stock", vix=18, daily_decay=-0.02)

    assert broad_index_result.sub_scores["volatility_decay"].raw_value == -0.001
    assert single_stock_result.sub_scores["volatility_decay"].raw_value == -0.02
    # -0.02 远超默认 decay_scale=0.005，应饱和到 -100；-0.001 未饱和
    assert single_stock_result.sub_scores["volatility_decay"].score == -100.0
    assert broad_index_result.sub_scores["volatility_decay"].score > -100.0


def test_target_type_vix_indicators_shared_regardless_of_target_type():
    # VIX 绝对水平/期限结构两个子指标两种 target_type 共用同一套计算逻辑，
    # 相同的 vix / vix_term_structure_ratio 输入应该得到相同的子分数。
    broad_index_result = score_leveraged_etf(target_type="broad_index", vix=12, vix_term_structure_ratio=0.9)
    single_stock_result = score_leveraged_etf(target_type="single_stock", vix=12, vix_term_structure_ratio=0.9)

    assert broad_index_result.sub_scores["vix_level"].score == single_stock_result.sub_scores["vix_level"].score
    assert (
        broad_index_result.sub_scores["vix_term_structure"].score
        == single_stock_result.sub_scores["vix_term_structure"].score
    )


def test_invalid_target_type_raises():
    with pytest.raises(ValueError):
        score_leveraged_etf(target_type="not_a_real_type", vix=10)


def test_valid_target_types_constant():
    assert VALID_TARGET_TYPES == {"broad_index", "single_stock"}


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
