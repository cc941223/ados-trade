"""engine.scoring.base 单元测试 —— linear_map / resolve_weights / aggregate_scores。

Confidence Score 用两个独立维度刻画多空证据：
    strength（绝对强度）  = (Bull+Bear) / 100
    agreement（方向一致性）= |Bull-Bear| / (Bull+Bear)，Bull=Bear=0 时约定为 0
    conviction            = strength * agreement
    confidence            = (completeness*0.5 + conviction*0.5) * 100

Bull+Bear 有理论上界 100（每个子指标 score∈[-100,100] 且权重非负），
下面同时验证这个不变量在越界输入时会被显式拒绝，而不是被 clip 悄悄掩盖。
"""
import pytest

from engine.scoring.base import SubScore, aggregate_scores, linear_map, resolve_weights


def test_linear_map_manual():
    # value=25 在 [0,100] 里占 25%，映射到 [-100,100] 应该是 -100 + 0.25*200 = -50
    assert linear_map(25, 0, 100, -100, 100) == -50

    # 中点映射到中点
    assert linear_map(50, 0, 100, -100, 100) == 0


def test_linear_map_reversed_output_range():
    # out_min/out_max 反向传入（数值越大 in 值，映射出的分数越低）
    # value=0(in_min) -> out_min=100；value=100(in_max) -> out_max=-100
    assert linear_map(0, 0, 100, 100, -100) == 100
    assert linear_map(100, 0, 100, 100, -100) == -100
    assert linear_map(50, 0, 100, 100, -100) == 0


def test_linear_map_clips_out_of_range():
    # value 超出 [in_min,in_max] 时应截断到对应边界
    assert linear_map(150, 0, 100, -100, 100) == 100
    assert linear_map(-50, 0, 100, -100, 100) == -100


def test_linear_map_none_passthrough():
    assert linear_map(None, 0, 100, -100, 100) is None


def test_linear_map_equal_bounds_raises():
    with pytest.raises(ValueError):
        linear_map(10, 5, 5, -100, 100)


def test_resolve_weights_default_and_override():
    defaults = {"a": 0.5, "b": 0.5}
    assert resolve_weights(defaults, None) == defaults
    assert resolve_weights(defaults, {"a": 0.7}) == {"a": 0.7, "b": 0.5}


def test_resolve_weights_unknown_key_raises():
    with pytest.raises(ValueError):
        resolve_weights({"a": 0.5}, {"unknown": 0.1})


def test_resolve_weights_negative_weight_raises():
    # 负权重会打破 aggregate_scores 依赖的 "Bull+Bear<=100" 不变量，必须在
    # 权重合并这一步就拒绝，不能留到聚合阶段才隐晦地出错。
    with pytest.raises(ValueError):
        resolve_weights({"a": 0.5, "b": 0.5}, {"a": -0.1})


def test_aggregate_scores_all_aligned_bullish_manual():
    # 3 个子指标权重各 1/3，全部看多分数 +100/+80/+60
    # Bull = (1/3*100 + 1/3*80 + 1/3*60) / 1 = 80
    # Bear = 0
    # strength = 80/100 = 0.8, agreement = |80-0|/80 = 1.0, conviction = 0.8
    # completeness = 1
    # confidence = (1*0.5 + 0.8*0.5)*100 = 90
    sub_scores = {
        "a": SubScore("a", 1.0, 100.0, 1 / 3),
        "b": SubScore("b", 1.0, 80.0, 1 / 3),
        "c": SubScore("c", 1.0, 60.0, 1 / 3),
    }
    result = aggregate_scores(sub_scores)

    assert abs(result.bull_score - 80.0) < 1e-9
    assert result.bear_score == 0.0
    assert abs(result.extra["strength"] - 0.8) < 1e-9
    assert abs(result.extra["agreement"] - 1.0) < 1e-9
    assert abs(result.confidence_score - 90.0) < 1e-9


def test_aggregate_scores_contradictory_lowers_confidence():
    # 4 个子指标权重各 0.25，两个看多(+100)两个看空(-100)：
    # Bull = 0.25*100 + 0.25*100 = 50；Bear = 0.25*100 + 0.25*100 = 50
    # strength = 100/100 = 1, agreement = |50-50|/100 = 0, conviction = 0
    # completeness = 1，confidence = (1*0.5 + 0*0.5)*100 = 50
    sub_scores = {
        "a": SubScore("a", 1.0, 100.0, 0.25),
        "b": SubScore("b", 1.0, 100.0, 0.25),
        "c": SubScore("c", -1.0, -100.0, 0.25),
        "d": SubScore("d", -1.0, -100.0, 0.25),
    }
    result = aggregate_scores(sub_scores)

    assert abs(result.bull_score - 50.0) < 1e-9
    assert abs(result.bear_score - 50.0) < 1e-9
    assert abs(result.confidence_score - 50.0) < 1e-9

    # 与全部同向的情况相比（此处全部饱和到±100，strength=1，与旧公式一致），
    # 矛盾情况下 confidence 明显更低
    aligned = aggregate_scores(
        {
            "a": SubScore("a", 1.0, 100.0, 0.25),
            "b": SubScore("b", 1.0, 100.0, 0.25),
            "c": SubScore("c", 1.0, 100.0, 0.25),
            "d": SubScore("d", 1.0, 100.0, 0.25),
        }
    )
    assert abs(aligned.confidence_score - 100.0) < 1e-9
    assert aligned.confidence_score - result.confidence_score > 30


def test_aggregate_scores_strong_vs_weak_disagreement_are_distinguished():
    # 用真实满足 Bull+Bear<=100 约束的数值验证：相对差距相同时，
    # "双强接近"(50 vs 45) 与 "双弱接近"(5 vs 4.5) 的 conviction 不应相同——
    # 这是本次修正要解决的问题（旧的 consistency=|Bull-Bear|/(Bull+Bear) 是
    # 尺度不变的比值，两者算出来完全一样，无法区分"证据打架"和"证据太弱"）。
    strong = aggregate_scores(
        {
            "a": SubScore("a", 1.0, 100.0, 0.5),
            "b": SubScore("b", -1.0, -90.0, 0.5),
        }
    )
    weak = aggregate_scores(
        {
            "a": SubScore("a", 1.0, 10.0, 0.5),
            "b": SubScore("b", -1.0, -9.0, 0.5),
        }
    )

    assert abs(strong.bull_score - 50.0) < 1e-9
    assert abs(strong.bear_score - 45.0) < 1e-9
    assert abs(weak.bull_score - 5.0) < 1e-9
    assert abs(weak.bear_score - 4.5) < 1e-9

    # 两者的相对差距（agreement）相同
    assert abs(strong.extra["agreement"] - weak.extra["agreement"]) < 1e-9
    # 但绝对强度（strength）不同，导致 conviction 与 confidence 不同
    assert strong.extra["strength"] > weak.extra["strength"]
    assert strong.confidence_score > weak.confidence_score
    assert abs(strong.extra["conviction"] - 0.05) < 1e-9
    assert abs(weak.extra["conviction"] - 0.005) < 1e-9


def test_aggregate_scores_missing_indicator_lowers_completeness():
    # 4 个子指标权重各 0.25，其中一个缺失（score=None）
    # 其余 3 个都看多且饱和 -> Bull=100（可用权重重新归一化后与全饱和等价）
    # strength=1, agreement=1, conviction=1
    # completeness = 0.75 / 1.0 = 0.75
    # confidence = (0.75*0.5 + 1*0.5)*100 = 87.5
    sub_scores = {
        "a": SubScore("a", 1.0, 100.0, 0.25),
        "b": SubScore("b", 1.0, 100.0, 0.25),
        "c": SubScore("c", 1.0, 100.0, 0.25),
        "d": SubScore("d", None, None, 0.25),
    }
    result = aggregate_scores(sub_scores)

    assert abs(result.extra["completeness"] - 0.75) < 1e-9
    assert abs(result.confidence_score - 87.5) < 1e-9


def test_aggregate_scores_no_signal_gives_zero_conviction():
    # 唯一的子指标分数为 0（无信号）：strength=0 -> conviction=0（不再像旧版本
    # 那样被约定为"完全一致"从而给出满分置信度贡献——没有证据不该被当作强证据）
    # completeness=1 -> confidence = (1*0.5 + 0*0.5)*100 = 50
    sub_scores = {"a": SubScore("a", 0.0, 0.0, 1.0)}
    result = aggregate_scores(sub_scores)

    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.extra["strength"] == 0.0
    assert result.extra["conviction"] == 0.0
    assert abs(result.confidence_score - 50.0) < 1e-9


def test_aggregate_scores_all_missing_returns_zero():
    sub_scores = {"a": SubScore("a", None, None, 1.0)}
    result = aggregate_scores(sub_scores)
    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.confidence_score == 0.0


def test_aggregate_scores_raises_when_signal_sum_exceeds_bound():
    # 直接构造一个越界 score（绕过 linear_map 的 clip），验证不变量检查会
    # 显式报错，而不是被 strength 悄悄 clip 掉。
    sub_scores = {"a": SubScore("a", 1.0, 150.0, 1.0)}
    with pytest.raises(ValueError):
        aggregate_scores(sub_scores)
