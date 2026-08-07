"""Put/Call Ratio 单元测试。"""
import math

from engine.indicators.options import put_call_ratio


def test_put_call_ratio_manual():
    # put=800, call=1000 => ratio = 0.8
    assert put_call_ratio(800, 1000) == 0.8


def test_put_call_ratio_call_zero_put_positive():
    assert put_call_ratio(100, 0) == float("inf")


def test_put_call_ratio_both_zero():
    assert math.isnan(put_call_ratio(0, 0))


def test_put_call_ratio_oi_style_input():
    # 同一函数也可用于 OI 口径
    assert put_call_ratio(15000, 20000) == 0.75
