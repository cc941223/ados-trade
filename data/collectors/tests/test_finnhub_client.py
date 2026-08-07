"""finnhub_client.py 单元测试 —— 验证请求逻辑（URL 拼接、参数）写对了，
不发真实网络请求（用注入的假 transport 函数代替 urllib）。
"""
import urllib.parse

from data.collectors.finnhub_client import FinnhubClient


def _capture_transport():
    """返回一个记录了最后一次请求 URL 的假 transport，以及取值函数。"""
    captured = {}

    def transport(url):
        captured["url"] = url
        return {"fake": "response"}

    return transport, captured


def test_get_earnings_calendar_builds_correct_url():
    transport, captured = _capture_transport()
    client = FinnhubClient(api_key="TESTKEY", transport=transport)

    client.get_earnings_calendar(from_date="2026-08-01", to_date="2026-08-31", symbol="AAPL")

    parsed = urllib.parse.urlparse(captured["url"])
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "finnhub.io"
    assert parsed.path == "/api/v1/calendar/earnings"
    assert query["from"] == ["2026-08-01"]
    assert query["to"] == ["2026-08-31"]
    assert query["symbol"] == ["AAPL"]
    assert query["token"] == ["TESTKEY"]


def test_get_earnings_calendar_without_symbol_omits_param():
    transport, captured = _capture_transport()
    client = FinnhubClient(api_key="TESTKEY", transport=transport)

    client.get_earnings_calendar(from_date="2026-08-01", to_date="2026-08-31")

    query = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
    assert "symbol" not in query


def test_get_upgrade_downgrade_builds_correct_url():
    transport, captured = _capture_transport()
    client = FinnhubClient(api_key="TESTKEY", transport=transport)

    client.get_upgrade_downgrade(symbol="AAPL", from_date="2026-01-01", to_date="2026-06-30")

    parsed = urllib.parse.urlparse(captured["url"])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/api/v1/stock/upgrade-downgrade"
    assert query["symbol"] == ["AAPL"]
    assert query["from"] == ["2026-01-01"]
    assert query["to"] == ["2026-06-30"]
    assert query["token"] == ["TESTKEY"]


def test_get_social_sentiment_builds_correct_url():
    transport, captured = _capture_transport()
    client = FinnhubClient(api_key="TESTKEY", transport=transport)

    client.get_social_sentiment(symbol="AAPL")

    parsed = urllib.parse.urlparse(captured["url"])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/api/v1/stock/social-sentiment"
    assert query["symbol"] == ["AAPL"]
    assert query["token"] == ["TESTKEY"]
    # 没传 from/to 时不应该出现在 query 里
    assert "from" not in query
    assert "to" not in query


def test_transport_return_value_passed_through():
    def fake_transport(url):
        return {"earningsCalendar": [{"symbol": "AAPL"}]}

    client = FinnhubClient(api_key="TESTKEY", transport=fake_transport)
    result = client.get_earnings_calendar(from_date="2026-08-01", to_date="2026-08-31")

    assert result == {"earningsCalendar": [{"symbol": "AAPL"}]}


def test_custom_transport_is_called_exactly_once_per_request():
    # 确认注入自定义 transport 后，请求确实经过它（不会真的走默认的
    # urllib 实现发起网络请求），而且每次方法调用只触发一次 transport。
    call_count = {"n": 0}

    def counting_transport(url):
        call_count["n"] += 1
        return {}

    client = FinnhubClient(api_key="TESTKEY", transport=counting_transport)
    client.get_social_sentiment(symbol="AAPL")

    assert call_count["n"] == 1
