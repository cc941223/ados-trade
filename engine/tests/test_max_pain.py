"""Max Pain 单元测试 —— 3 个行使价手算对照。"""
from engine.indicators.options import max_pain


def test_max_pain_manual_three_strikes():
    # strikes = [95, 100, 105]
    # call_oi = [100, 50, 0]
    # put_oi  = [0, 50, 100]
    #
    # 手算 payout：
    #  K=95:  call=0                         put=5*50+10*100=1250      total=1250
    #  K=100: call=5*100=500                 put=5*100=500             total=1000
    #  K=105: call=10*100+5*50=1250          put=0                      total=1250
    #
    # payout 最小值在 K=100，即 Max Pain = 100
    strikes = [95, 100, 105]
    call_oi = [100, 50, 0]
    put_oi = [0, 50, 100]

    result = max_pain(strikes, call_oi, put_oi)
    assert result == 100


def test_max_pain_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        max_pain([95, 100], [1, 2, 3], [1, 2])


def test_max_pain_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        max_pain([], [], [])
