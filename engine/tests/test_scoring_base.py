"""engine.scoring.base 单元测试 —— linear_map / resolve_weights / aggregate_scores。"""
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


def test_aggregate_scores_all_aligned_bullish_manual():
    # 3 个子指标权重各 1/3，全部看多分数 +100/+80/+60
    # Bull = (1/3*100 + 1/3*80 + 1/3*60) / 1 = 80
    # Bear = 0
    # completeness = 1, consistency = |80-0|/80 = 1, confidence = 100
    sub_scores = {
        "a": SubScore("a", 1.0, 100.0, 1 / 3),
        "b": SubScore("b", 1.0, 80.0, 1 / 3),
        "c": SubScore("c", 1.0, 60.0, 1 / 3),
    }
    result = aggregate_scores(sub_scores)

    assert abs(result.bull_score - 80.0) < 1e-9
    assert result.bear_score == 0.0
    assert abs(result.confidence_score - 100.0) < 1e-9


def test_aggregate_scores_contradictory_lowers_confidence():
    # 4 个子指标权重各 0.25，两个看多(+100)两个看空(-100)：
    # Bull = 0.25*100 + 0.25*100 = 50；Bear = 0.25*100 + 0.25*100 = 50
    # completeness = 1，consistency = |50-50|/100 = 0，confidence = 50
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

    # 与全部同向的情况相比，矛盾情况下 confidence 明显更低
    aligned = aggregate_scores(
        {
            "a": SubScore("a", 1.0, 100.0, 0.25),
            "b": SubScore("b", 1.0, 100.0, 0.25),
            "c": SubScore("c", 1.0, 100.0, 0.25),
            "d": SubScore("d", 1.0, 100.0, 0.25),
        }
    )
    assert aligned.confidence_score - result.confidence_score > 30


def test_aggregate_scores_missing_indicator_lowers_completeness():
    # 4 个子指标权重各 0.25，其中一个缺失（score=None）
    # completeness = 0.75 / 1.0 = 0.75；其余 3 个都看多 -> consistency = 1
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


def test_aggregate_scores_no_signal_treated_as_consistent():
    # 唯一的子指标分数为 0（无信号），约定不算矛盾：consistency = 1
    sub_scores = {"a": SubScore("a", 0.0, 0.0, 1.0)}
    result = aggregate_scores(sub_scores)

    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.extra["consistency"] == 1.0


def test_aggregate_scores_all_missing_returns_zero():
    sub_scores = {"a": SubScore("a", None, None, 1.0)}
    result = aggregate_scores(sub_scores)
    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.confidence_score == 0.0
