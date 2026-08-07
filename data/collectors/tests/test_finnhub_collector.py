"""finnhub_collector.py 单元测试 —— 用假 JSON 响应（模拟 Finnhub 真实
返回格式）验证解析结果跟 schema.sql 字段对得上，覆盖正常情况和缺字段的
情况。全程通过注入 transport 避免真实网络请求。
"""
from datetime import date, datetime, timezone

from data.collectors.finnhub_client import FinnhubClient
from data.collectors.finnhub_collector import (
    collect_analyst_ratings,
    collect_earnings_calendar,
    collect_social_sentiment,
)


def _client_with_response(response: object) -> FinnhubClient:
    return FinnhubClient(api_key="TESTKEY", transport=lambda url: response)


# ---------------------------------------------------------------------------
# earnings_calendar
# ---------------------------------------------------------------------------


def test_collect_earnings_calendar_normal_case():
    client = _client_with_response(
        {
            "earningsCalendar": [
                {
                    "date": "2026-08-20",
                    "epsActual": 2.18,
                    "epsEstimate": 2.10,
                    "hour": "amc",
                    "quarter": 3,
                    "revenueActual": 119_575_000_000,
                    "revenueEstimate": 117_910_000_000,
                    "symbol": "AAPL",
                    "year": 2026,
                }
            ]
        }
    )

    rows = collect_earnings_calendar(client, from_date="2026-08-01", to_date="2026-08-31")

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "symbol", "report_date", "time_of_day", "is_confirmed", "eps_estimate", "eps_actual",
        "eps_surprise_pct", "revenue_estimate", "revenue_actual", "revenue_surprise_pct", "actual_move_pct",
    }
    assert row["symbol"] == "AAPL"
    assert row["report_date"] == date(2026, 8, 20)
    assert row["time_of_day"] == "AMC"
    assert row["is_confirmed"] is None
    # eps_surprise_pct = (2.18-2.10)/2.10*100 = 3.809523...
    assert abs(row["eps_surprise_pct"] - (2.18 - 2.10) / 2.10 * 100) < 1e-9
    # revenue_surprise_pct = (119575000000-117910000000)/117910000000*100
    expected_rev_surprise = (119_575_000_000 - 117_910_000_000) / 117_910_000_000 * 100
    assert abs(row["revenue_surprise_pct"] - expected_rev_surprise) < 1e-9
    assert row["actual_move_pct"] is None


def test_collect_earnings_calendar_bmo_and_unknown_hour():
    client = _client_with_response(
        {
            "earningsCalendar": [
                {"date": "2026-08-20", "hour": "bmo", "symbol": "AAPL"},
                {"date": "2026-08-21", "hour": "dmh", "symbol": "MSFT"},  # 盘中，未定义取值
                {"date": "2026-08-22", "hour": "", "symbol": "GOOG"},  # 未知
            ]
        }
    )

    rows = collect_earnings_calendar(client, from_date="2026-08-01", to_date="2026-08-31")

    assert rows[0]["time_of_day"] == "BMO"
    assert rows[1]["time_of_day"] is None
    assert rows[2]["time_of_day"] is None


def test_collect_earnings_calendar_missing_eps_fields_no_surprise_computed():
    # 只有 estimate 没有 actual（还没到财报日）——不应该算出 surprise_pct
    client = _client_with_response(
        {"earningsCalendar": [{"date": "2026-09-01", "symbol": "AAPL", "epsEstimate": 2.20}]}
    )

    rows = collect_earnings_calendar(client, from_date="2026-09-01", to_date="2026-09-30")

    assert rows[0]["eps_estimate"] == 2.20
    assert rows[0]["eps_actual"] is None
    assert rows[0]["eps_surprise_pct"] is None


def test_collect_earnings_calendar_zero_estimate_avoids_division_by_zero():
    client = _client_with_response(
        {"earningsCalendar": [{"date": "2026-09-01", "symbol": "AAPL", "epsEstimate": 0, "epsActual": 0.05}]}
    )
    rows = collect_earnings_calendar(client, from_date="2026-09-01", to_date="2026-09-30")
    assert rows[0]["eps_surprise_pct"] is None


def test_collect_earnings_calendar_empty_response():
    client = _client_with_response({"earningsCalendar": []})
    rows = collect_earnings_calendar(client, from_date="2026-09-01", to_date="2026-09-30")
    assert rows == []


def test_collect_earnings_calendar_none_response():
    client = _client_with_response(None)
    rows = collect_earnings_calendar(client, from_date="2026-09-01", to_date="2026-09-30")
    assert rows == []


# ---------------------------------------------------------------------------
# analyst_ratings
# ---------------------------------------------------------------------------


def test_collect_analyst_ratings_normal_case():
    client = _client_with_response(
        [
            {
                "symbol": "AAPL",
                "gradeTime": 1611878400,  # 2021-01-29T00:00:00Z
                "company": "Wedbush",
                "fromGrade": "Neutral",
                "toGrade": "Outperform",
                "action": "up",
            }
        ]
    )

    rows = collect_analyst_ratings(client, symbol="AAPL")

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"symbol", "ts", "firm", "rating", "price_target"}
    assert row["symbol"] == "AAPL"
    assert row["ts"] == datetime(2021, 1, 29, 0, 0, 0, tzinfo=timezone.utc)
    assert row["firm"] == "Wedbush"
    assert row["rating"] == "Outperform"
    # price_target 见 collect_analyst_ratings 文档说明的已知缺口
    assert row["price_target"] is None


def test_collect_analyst_ratings_missing_grade_time():
    client = _client_with_response([{"symbol": "AAPL", "company": "Wedbush", "toGrade": "Outperform"}])
    rows = collect_analyst_ratings(client, symbol="AAPL")
    assert rows[0]["ts"] is None


def test_collect_analyst_ratings_empty_response():
    client = _client_with_response([])
    rows = collect_analyst_ratings(client, symbol="AAPL")
    assert rows == []


# ---------------------------------------------------------------------------
# social_sentiment
# ---------------------------------------------------------------------------


def test_collect_social_sentiment_normal_case():
    client = _client_with_response(
        {
            "symbol": "AAPL",
            "reddit": [
                {
                    "atTime": "2026-08-06 00:00:00",
                    "mention": 10,
                    "positiveScore": 0.6,
                    "negativeScore": 0.4,
                    "positiveMention": 6,
                    "negativeMention": 4,
                    "score": 0.2,
                }
            ],
            "twitter": [
                {
                    "atTime": "2026-08-06 00:00:00",
                    "mention": 50,
                    "positiveScore": 0.55,
                    "negativeScore": 0.45,
                    "score": 0.1,
                }
            ],
        }
    )

    rows = collect_social_sentiment(client, symbol="AAPL")

    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"reddit", "twitter"}

    reddit_row = next(r for r in rows if r["source"] == "reddit")
    assert set(reddit_row) == {"symbol", "ts", "source", "mention_count", "positive_score", "negative_score", "mspr"}
    assert reddit_row["symbol"] == "AAPL"
    assert reddit_row["ts"] == datetime(2026, 8, 6, 0, 0, 0, tzinfo=timezone.utc)
    assert reddit_row["mention_count"] == 10
    assert reddit_row["positive_score"] == 0.6
    assert reddit_row["negative_score"] == 0.4
    assert reddit_row["mspr"] == 0.2

    twitter_row = next(r for r in rows if r["source"] == "twitter")
    assert twitter_row["mention_count"] == 50
    assert twitter_row["mspr"] == 0.1


def test_collect_social_sentiment_missing_twitter_key_only_returns_reddit():
    client = _client_with_response(
        {"symbol": "AAPL", "reddit": [{"atTime": "2026-08-06 00:00:00", "mention": 5, "score": 0.1}]}
    )
    rows = collect_social_sentiment(client, symbol="AAPL")
    assert len(rows) == 1
    assert rows[0]["source"] == "reddit"


def test_collect_social_sentiment_empty_response():
    client = _client_with_response({})
    rows = collect_social_sentiment(client, symbol="AAPL")
    assert rows == []


def test_collect_social_sentiment_entry_missing_optional_scores():
    client = _client_with_response(
        {"symbol": "AAPL", "reddit": [{"atTime": "2026-08-06 00:00:00", "mention": 5}]}  # 没有 score/positiveScore
    )
    rows = collect_social_sentiment(client, symbol="AAPL")
    assert rows[0]["mspr"] is None
    assert rows[0]["positive_score"] is None
    assert rows[0]["mention_count"] == 5
