"""IBKR 券商数据读取实现。

默认路径是 `IBInsyncClient`（ib_insync + IB Gateway Docker，TWS API 协议，
见 `client.py`）。旧的 IBeam + CPAPI 实现（`IBKRClient` + `collect_*`
解析函数）保留在 `brokers.ibkr.legacy_ibeam`，仅供回退参考，不在这里
默认导出——需要用它时显式 `from brokers.ibkr.legacy_ibeam import ...`。
"""
from .client import IBInsyncClient

__all__ = [
    "IBInsyncClient",
]
