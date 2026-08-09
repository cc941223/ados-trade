"""数据采集层：把外部数据源（Finnhub）的原始返回，解析成跟
`data/schema.sql` 表字段一一对应的 dict 列表，供以后接一层简单的写库
逻辑直接使用。本包不包含任何数据库读写代码。

⚠️ 结构重构说明：IBKR 相关的 `ibkr_client.py`/`ibkr_collector.py`（以及
`MockIBKRClient`）已经搬到 `brokers/`（`brokers/ibkr/`、`brokers/mock/`）
——因为 IBKR 是"券商"数据源，需要走统一的 `brokers.base.BrokerClient`
接口，跟这里 Finnhub（财报/分析师评级/社交情绪，不是券商概念）分开管理。
本包现在只保留 Finnhub 部分，未做任何改动。
"""
from .finnhub_client import FinnhubClient
from .finnhub_collector import collect_analyst_ratings, collect_earnings_calendar, collect_social_sentiment

__all__ = [
    "FinnhubClient",
    "collect_earnings_calendar",
    "collect_analyst_ratings",
    "collect_social_sentiment",
]
