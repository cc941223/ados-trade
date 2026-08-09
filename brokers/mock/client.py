"""测试/开发用的假券商客户端，不发任何网络请求。

⚠️ 结构重构说明：`MockIBKRClient` 原来跟 `IBKRClient` 定义在同一个文件
（`data/collectors/ibkr_client.py`）里，本次按你的要求把它搬到独立的
`brokers/mock/` 目录，跟 `brokers/ibkr/`（真实 IBKR 实现）分开放。

⚠️ 设计决策（未在需求里明确写出，这里主动标出来）：`MockIBKRClient`
仍然继承自 `brokers.ibkr.client.IBKRClient`，不是直接继承
`brokers.base.BrokerClient` 重新实现一遍。这样它可以直接复用
`IBKRClient` 已经实现好的 `get_stock_ohlcv`/`get_option_chain`/
`get_option_snapshot`/`get_iv_percentile`（这四个方法内部只是把"原始格式
返回值"解析成 schema 表行，解析逻辑跟真实 IBKR 客户端应该完全一样，没有
理由在 Mock 里重复写一遍），自己只需要覆盖四个返回"原始格式"数据的方法
（`get_price_history`/`_get_option_chain_raw`/`get_option_market_data`/
`_get_iv_percentile_raw`），跟重构前的继承关系完全一致，只是现在两个类
分别放在 `brokers/ibkr/` 和 `brokers/mock/` 两个目录，之间有一个跨目录的
import。如果你更想要 `MockIBKRClient` 完全独立于 `IBKRClient`（不共享任何
实现，直接对着 `BrokerClient` 重新写四个方法，牺牲一点重复代码换取两者
彻底解耦），告诉我，这一行继承关系可以改。

默认数据是硬编码的固定值（不是随机生成），方便测试里手算核对解析结果；
也可以在构造时传 `fixture` 覆盖，测试"缺字段"等边界情况。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from ..ibkr.client import IBKRClient


class MockIBKRClient(IBKRClient):
    """测试/开发用的假实现，返回构造好的固定假数据，不发任何网络请求。"""

    def __init__(self, fixture: Optional[dict] = None):
        self._fixture = fixture or {}

    def get_price_history(
        self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[dict]:
        if "price_history" in self._fixture:
            return self._fixture["price_history"]
        # 假想的 CPAPI 历史数据返回格式：t=epoch毫秒，o/h/l/c=开高低收，v=成交量
        return [
            {"t": 1735689600000, "o": 190.0, "h": 192.5, "l": 189.0, "c": 191.2, "v": 45_000_000},
            {"t": 1735776000000, "o": 191.5, "h": 193.0, "l": 190.5, "c": 192.8, "v": 38_000_000},
        ]

    def _get_option_chain_raw(self, underlying: str, expiration_date: date) -> list[dict]:
        if "option_chain" in self._fixture:
            return self._fixture["option_chain"]
        return [
            {"conid": 500001, "strike": 190.0, "right": "C"},
            {"conid": 500002, "strike": 195.0, "right": "C"},
            {"conid": 500003, "strike": 190.0, "right": "P"},
            {"conid": 500004, "strike": 195.0, "right": "P"},
        ]

    def get_option_market_data(self, contract_ids: list[int]) -> list[dict]:
        if "option_market_data" in self._fixture:
            return self._fixture["option_market_data"]
        fake_snapshots = {
            500001: {"conid": 500001, "last": 5.20, "bid": 5.10, "ask": 5.30, "volume": 1200,
                     "open_interest": 4500, "implied_vol": 0.28, "delta": 0.62, "gamma": 0.04,
                     "theta": -0.05, "vega": 0.12},
            500002: {"conid": 500002, "last": 2.10, "bid": 2.00, "ask": 2.20, "volume": 800,
                     "open_interest": 3000, "implied_vol": 0.26, "delta": 0.35, "gamma": 0.05,
                     "theta": -0.04, "vega": 0.10},
            500003: {"conid": 500003, "last": 4.80, "bid": 4.70, "ask": 4.90, "volume": 950,
                     "open_interest": 4100, "implied_vol": 0.29, "delta": -0.38, "gamma": 0.04,
                     "theta": -0.05, "vega": 0.12},
            500004: {"conid": 500004, "last": 7.50, "bid": 7.35, "ask": 7.65, "volume": 600,
                     "open_interest": 2200, "implied_vol": 0.27, "delta": -0.65, "gamma": 0.05,
                     "theta": -0.04, "vega": 0.10},
        }
        return [fake_snapshots[cid] for cid in contract_ids if cid in fake_snapshots]

    def _get_iv_percentile_raw(self, symbol: str) -> dict:
        if "iv_percentile" in self._fixture:
            return self._fixture["iv_percentile"]
        return {"13w": 45.2, "26w": 52.8, "52w": 61.5}
