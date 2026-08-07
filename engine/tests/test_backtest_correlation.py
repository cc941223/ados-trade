"""compute_information_coefficient() 单元测试 —— 手算/独立库对照。

Pearson 相关系数用 `scipy.stats.pearsonr`（已经在 test_black_scholes.py
等测试里作为可信第三方库使用过）独立算出期望值核对；Spearman 同理用
`scipy.stats.spearmanr` 核对。
"""
import pytest
from scipy.stats import pearsonr, spearmanr

from engine.backtest.correlation import MIN_RELIABLE_SAMPLE_SIZE, compute_information_coefficient


def test_pearson_correlation_matches_scipy():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    expected, _ = pearsonr(x, y)

    result = compute_information_coefficient(x, y, method="pearson")

    assert abs(result.correlation_ic - expected) < 1e-9
    assert result.sample_size == 5
    assert result.is_reliable is False  # 5 < 30


def test_spearman_correlation_matches_scipy():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    expected, _ = spearmanr(x, y)

    result = compute_information_coefficient(x, y, method="spearman")

    assert abs(result.correlation_ic - expected) < 1e-9


def test_default_method_is_spearman_not_pearson():
    # 默认不传 method 时应该走 Spearman（跟量化圈"IC"的行业惯例对齐），
    # 不是 Pearson——用一组两种方法算出来数值不同的数据（有并列值，
    # pearson≈0.7746，spearman≈0.7379）验证默认值确实是 spearman，
    # 不只是碰巧两种方法结果一样导致测试测不出区别。
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    pearson_expected, _ = pearsonr(x, y)
    spearman_expected, _ = spearmanr(x, y)
    assert abs(pearson_expected - spearman_expected) > 1e-3  # 确认两种方法在这组数据上确实有区别

    default_result = compute_information_coefficient(x, y)  # 不传 method

    assert abs(default_result.correlation_ic - spearman_expected) < 1e-9
    assert abs(default_result.correlation_ic - pearson_expected) > 1e-3


def test_perfect_positive_correlation():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    result = compute_information_coefficient(x, y)
    assert abs(result.correlation_ic - 1.0) < 1e-9


def test_perfect_negative_correlation():
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    result = compute_information_coefficient(x, y)
    assert abs(result.correlation_ic - (-1.0)) < 1e-9


def test_sample_size_below_threshold_is_not_reliable():
    # 6.3 节硬性要求：样本量低于 30 次时标注 is_reliable=False
    x = list(range(29))
    y = [v * 2 for v in x]
    result = compute_information_coefficient(x, y)

    assert result.sample_size == 29
    assert result.is_reliable is False
    # 注意：即使样本不足，correlation_ic 仍然照算，只是标注不可采信，
    # 不是把数值也抹掉——is_reliable 和 correlation_ic 是两件独立的事。
    assert result.correlation_ic is not None


def test_sample_size_at_threshold_is_reliable():
    x = list(range(30))
    y = [v * 2 + 1 for v in x]
    result = compute_information_coefficient(x, y)

    assert result.sample_size == 30
    assert result.is_reliable is True


def test_sample_size_above_threshold_is_reliable():
    x = list(range(50))
    y = [v * -1 for v in x]
    result = compute_information_coefficient(x, y)
    assert result.is_reliable is True


def test_custom_min_sample_size():
    x = list(range(10))
    y = list(range(10))
    result = compute_information_coefficient(x, y, min_sample_size=5)
    assert result.is_reliable is True  # 10 >= 5


def test_insufficient_data_for_correlation_returns_none():
    # 只有 1 个观测点，相关系数没有数学定义
    result = compute_information_coefficient([1.0], [0.05])
    assert result.correlation_ic is None
    assert result.sample_size == 1
    assert result.is_reliable is False


def test_zero_variance_indicator_returns_none_not_zero():
    # 指标值全部相同（方差为0），相关系数无定义，应该是 None 不是 0
    x = [5.0] * 10
    y = list(range(10))
    result = compute_information_coefficient(x, y)
    assert result.correlation_ic is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_information_coefficient([1, 2, 3], [1, 2])


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        compute_information_coefficient([1, 2, 3], [1, 2, 3], method="kendall")


def test_indicator_name_and_scenario_propagate():
    result = compute_information_coefficient(
        [1, 2, 3, 4], [4, 3, 2, 1], indicator_name="iv_skew", scenario="options"
    )
    assert result.indicator_name == "iv_skew"
    assert result.scenario == "options"


def test_default_min_reliable_sample_size_constant():
    assert MIN_RELIABLE_SAMPLE_SIZE == 30
