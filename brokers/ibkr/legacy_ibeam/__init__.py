"""⚠️ 已废弃：IBeam + Client Portal Web API（CPAPI）实现，仅供回退参考。

M0 阶段的默认连接方式已经换成 `brokers.ibkr.client.IBInsyncClient`
（ib_insync + IB Gateway Docker）。这个子包完整保留旧实现（不删除），
但不会被 `docker-compose.yml` 启动，也不是 `brokers.ibkr` 的默认导出。
"""
from .client import IBKRClient
from .collector import collect_iv_percentile, collect_option_chain_and_snapshot, collect_stock_ohlcv

__all__ = [
    "IBKRClient",
    "collect_stock_ohlcv",
    "collect_option_chain_and_snapshot",
    "collect_iv_percentile",
]
