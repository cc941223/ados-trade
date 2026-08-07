"""数据采集层：把外部数据源（IBKR CPAPI / Finnhub）的原始返回，解析成跟
`data/schema.sql` 表字段一一对应的 dict 列表，供以后接一层简单的写库
逻辑直接使用。本包不包含任何数据库读写代码。
"""
from .finnhub_client import FinnhubClient
from .finnhub_collector import collect_analyst_ratings, collect_earnings_calendar, collect_social_sentiment
from .ibkr_client import IBKRClient, MockIBKRClient
from .ibkr_collector import collect_iv_percentile, collect_option_chain_and_snapshot, collect_stock_ohlcv

__all__ = [
    "IBKRClient",
    "MockIBKRClient",
    "collect_stock_ohlcv",
    "collect_option_chain_and_snapshot",
    "collect_iv_percentile",
    "FinnhubClient",
    "collect_earnings_calendar",
    "collect_analyst_ratings",
    "collect_social_sentiment",
]
