"""Finnhub 数据采集：财报日历（含历史意外幅度）/ 分析师评级 / 社交媒体情绪。

跟 `ibkr_collector.py` 同样的模式：`collect_*` 函数调用 `FinnhubClient`
拿原始 JSON，再用 `_parse_*` 函数转换成跟 `data/schema.sql` 表字段一一
对应的 dict 列表。
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from .finnhub_client import FinnhubClient

#: Finnhub `hour` 字段到 `earnings_calendar.time_of_day` 取值的映射。
#: Finnhub 除了 "bmo"（盘前）/"amc"（盘后）之外，还会返回 "dmh"（盘中）
#: 或空字符串（未知），这两种情况 `time_of_day` 映射为 None，跟 schema
#: 注释"'BMO'/'AMC'/NULL(未知)"的取值范围对齐（只定义了盘前/盘后/未知
#: 三种，没有"盘中"这个第四种取值，所以盘中也归到 None）。
_HOUR_TO_TIME_OF_DAY = {"bmo": "BMO", "amc": "AMC"}


def _surprise_pct(actual: Optional[float], estimate: Optional[float]) -> Optional[float]:
    """(实际值-预期值)/|预期值| * 100，两者任一缺失或预期值为 0 时返回 None。

    Finnhub 的财报日历只给 actual/estimate 两个原始数值，不直接给百分比，
    这里做的是纯算术换算（跟单位无关的相对偏离百分比），不是新指标计算。
    """
    if actual is None or estimate is None or estimate == 0:
        return None
    return (actual - estimate) / abs(estimate) * 100


def collect_earnings_calendar(
    client: FinnhubClient, from_date: str, to_date: str, symbol: Optional[str] = None
) -> list[dict]:
    """采集财报日历 + 历史意外幅度，对应 `earnings_calendar` 表。

    Parameters
    ----------
    client : `FinnhubClient`。
    from_date, to_date : 查询的日期范围，"YYYY-MM-DD"。
    symbol : 只查某个标的时传，不传则返回该时间范围内所有标的的财报。

    Returns
    -------
    list[dict]，key 是 `earnings_calendar` 的列名：symbol/report_date/
    time_of_day/is_confirmed/eps_estimate/eps_actual/eps_surprise_pct/
    revenue_estimate/revenue_actual/revenue_surprise_pct/actual_move_pct。

    ⚠️ 两个字段本函数无法填：
    - `is_confirmed`：Finnhub 免费层的财报日历接口不区分"官方确认"还是
      "第三方预估"日期，这里统一留 `None`（不是 `False`——`False` 意味着
      "确认过、答案是没确认"，`None` 才是"不知道"，两者含义不同）。要拿到
      真正的确认状态，需要另外的数据源或 Finnhub 付费层，超出这个采集
      函数的能力范围。
    - `actual_move_pct`：财报后次日实际涨跌幅，需要拿这个 `report_date`
      去 `stock_ohlcv` 表查对应日期前后的收盘价再算涨跌幅，这是跨表联合
      计算，不是"调用一个 API 解析一下返回"能做的事，留给以后专门写一个
      "财报日历 + OHLCV 关联计算"的批处理脚本去补，这里统一留 `None`。
    """
    raw = client.get_earnings_calendar(from_date, to_date, symbol)
    items = (raw or {}).get("earningsCalendar", [])
    return [_parse_earnings_item(item) for item in items]


def _parse_earnings_item(item: dict) -> dict:
    eps_actual = item.get("epsActual")
    eps_estimate = item.get("epsEstimate")
    revenue_actual = item.get("revenueActual")
    revenue_estimate = item.get("revenueEstimate")

    return {
        "symbol": item.get("symbol"),
        "report_date": date.fromisoformat(item["date"]) if item.get("date") else None,
        "time_of_day": _HOUR_TO_TIME_OF_DAY.get(item.get("hour")),
        "is_confirmed": None,  # 见 collect_earnings_calendar 文档里的说明
        "eps_estimate": eps_estimate,
        "eps_actual": eps_actual,
        "eps_surprise_pct": _surprise_pct(eps_actual, eps_estimate),
        "revenue_estimate": revenue_estimate,
        "revenue_actual": revenue_actual,
        "revenue_surprise_pct": _surprise_pct(revenue_actual, revenue_estimate),
        "actual_move_pct": None,  # 见 collect_earnings_calendar 文档里的说明
    }


def collect_analyst_ratings(
    client: FinnhubClient, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
) -> list[dict]:
    """采集分析师评级变动，对应 `analyst_ratings` 表。

    Returns
    -------
    list[dict]，key 是 `analyst_ratings` 的列名：symbol/ts/firm/rating/
    price_target。

    ⚠️ `price_target` 恒为 `None`——见 `FinnhubClient.get_upgrade_downgrade`
    文档里的说明，评级变动端点不带目标价，Finnhub 的目标价是另一个聚合
    统计端点，跟"单家机构一条记录"的表设计对不上，这个缺口需要你确认
    怎么处理，不是本函数能自己决定的事。
    """
    raw = client.get_upgrade_downgrade(symbol, from_date, to_date)
    items = raw or []
    return [_parse_analyst_rating_item(item) for item in items]


def _parse_analyst_rating_item(item: dict) -> dict:
    grade_time = item.get("gradeTime")
    return {
        "symbol": item.get("symbol"),
        "ts": datetime.fromtimestamp(grade_time, tz=timezone.utc) if grade_time is not None else None,
        "firm": item.get("company"),
        "rating": item.get("toGrade"),
        "price_target": None,
    }


def collect_social_sentiment(
    client: FinnhubClient, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
) -> list[dict]:
    """采集 Reddit/Twitter 社交媒体情绪，对应 `social_sentiment` 表。

    Returns
    -------
    list[dict]，key 是 `social_sentiment` 的列名：symbol/ts/source/
    mention_count/positive_score/negative_score/mspr。Reddit 和 Twitter
    各自的时间点分别展开成独立的行（不是每个时间点合并成一行），跟
    schema 的主键 `(symbol, ts, source)` 设计一致——同一个 symbol 同一个
    ts，reddit 和 twitter 是两条不同的记录。
    """
    raw = client.get_social_sentiment(symbol, from_date, to_date) or {}
    rows = []
    for source in ("reddit", "twitter"):
        for entry in raw.get(source, []) or []:
            rows.append(_parse_social_sentiment_entry(symbol, source, entry))
    return rows


def _parse_social_sentiment_entry(symbol: str, source: str, entry: dict) -> dict:
    at_time = entry.get("atTime")
    ts = None
    if at_time:
        ts = datetime.strptime(at_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    return {
        "symbol": symbol,
        "ts": ts,
        "source": source,
        "mention_count": entry.get("mention"),
        "positive_score": entry.get("positiveScore"),
        "negative_score": entry.get("negativeScore"),
        "mspr": entry.get("score"),
    }
