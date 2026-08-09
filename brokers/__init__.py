"""券商数据读取层：统一接口 + 具体实现（真实 IBKR / 测试用 Mock）。

跟 `data/collectors` 的分工：Finnhub 那套（财报/分析师评级/社交情绪）不是
"券商"概念，仍然留在 `data/collectors/finnhub_*.py`，不纳入这里的抽象。
"""
from .base import BrokerClient

__all__ = ["BrokerClient"]
