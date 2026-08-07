"""VIX 期限结构比值 单元测试 —— 手算对照。"""
import pytest

from engine.indicators.leveraged import vix_term_structure_ratio


def test_vix_term_structure_contango_manual():
    # front(近月)=18, back(远月)=20 -> ratio = 0.9 < 1，正向期限结构（contango）
    result = vix_term_structure_ratio(18, 20)
    assert abs(result["ratio"] - 0.9) < 1e-9
    assert result["structure"] == "contango"


def test_vix_term_structure_backwardation_manual():
    # front(近月)=25, back(远月)=20 -> ratio = 1.25 > 1，逆向期限结构（backwardation）
    result = vix_term_structure_ratio(25, 20)
    assert abs(result["ratio"] - 1.25) < 1e-9
    assert result["structure"] == "backwardation"


def test_vix_term_structure_back_zero_raises():
    with pytest.raises(ValueError):
        vix_term_structure_ratio(20, 0)
