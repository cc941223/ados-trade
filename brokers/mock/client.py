"""测试/开发用的假券商客户端，不发任何网络请求。

`MockIBKRClient` 直接实现 `brokers.base.BrokerClient`，跟
`brokers.ibkr.client.IBKRClient` 之间没有继承关系、不导入对方任何代码
（也不导入 `brokers/ibkr/collector.py` 的 `collect_*`/`_parse_*` 函数）——
两边各自独立地把"假想的 CPAPI 原始返回格式"解析成 `data/schema.sql`
表结构的行，字段映射规则重复了一份，是刻意的：保证这个测试替身作为独立
个体存在，不会因为真实 IBKR 实现那边的内部结构调整而被动受影响。

分两组方法：

- 四个"原始 CPAPI 格式"方法（`get_price_history`/`_get_option_chain_raw`/
  `get_option_market_data`/`_get_iv_percentile_raw`）：方法名跟
  `IBKRClient` 保持一致（鸭子类型，不是继承），因为
  `brokers/ibkr/collector.py` 里的 `collect_*` 函数是按这几个方法名去调用
  客户端的，`MockIBKRClient` 需要能替代真实 `IBKRClient` 传给那些函数、
  被现有测试（`brokers/tests/test_ibkr_collector.py`）复用。
- 四个 `BrokerClient` 统一接口方法（`get_stock_ohlcv`/`get_option_chain`/
  `get_option_snapshot`/`get_iv_percentile`）：内部调用的是上面这组
  "自己的"原始格式方法，再自己解析成 schema 行——解析这一步是独立写的，
  没有调用 `collector.py` 那边的 `_parse_*` 函数。

默认数据是硬编码的固定值（不是随机生成），方便测试里手算核对解析结果；
也可以在构造时传 `fixture` 覆盖，测试"缺字段"等边界情况。
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from ..base import BrokerClient


class MockIBKRClient(BrokerClient):
    """测试/开发用的假实现，返回构造好的固定假数据，不发任何网络请求。"""

    def __init__(self, fixture: Optional[dict] = None):
        self._fixture = fixture or {}

    # ------------------------------------------------------------------
    # 原始 CPAPI 格式方法（跟 IBKRClient 同名，鸭子类型，不是继承）
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # brokers.base.BrokerClient 统一接口实现——调用上面"自己的"原始格式
    # 方法，再独立解析成 schema 行（跟 collector.py 的 _parse_* 函数不是
    # 同一份代码，字段映射规则重复了一份）。
    # ------------------------------------------------------------------

    def get_stock_ohlcv(
        self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[dict]:
        raw_bars = self.get_price_history(symbol, timeframe, start, end)
        return [
            {
                "symbol": symbol,
                "ts": datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc),
                "timeframe": timeframe,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
            }
            for bar in raw_bars
        ]

    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
        raw_contracts = self._get_option_chain_raw(underlying, expiration_date)
        return [
            {
                "contract_id": contract["conid"],
                "underlying": underlying,
                "expiration_date": expiration_date,
                "strike": contract.get("strike"),
                "right": contract.get("right"),
            }
            for contract in raw_contracts
        ]

    def get_option_snapshot(self, contract_ids: list[int]) -> list[dict]:
        raw_snapshots = self.get_option_market_data(contract_ids)
        snapshot_ts = datetime.now(timezone.utc)
        return [
            {
                "contract_id": snapshot["conid"],
                "ts": snapshot_ts,
                "last_price": snapshot.get("last"),
                "bid": snapshot.get("bid"),
                "ask": snapshot.get("ask"),
                "volume": snapshot.get("volume"),
                "open_interest": snapshot.get("open_interest"),
                # Greeks 缺失是正常情况——深度虚值/流动性差的合约，IBKR 有时
                # 不返回 Greeks 字段，缺了就是 None，不是 0（0 是"确实算出来
                # 是0"，跟"没返回"含义不同，交给 Engine 层后续用 IV 反推补上，
                # 见 schema.sql 里 option_snapshot.delta 列的注释）。
                "implied_vol": snapshot.get("implied_vol"),
                "delta": snapshot.get("delta"),
                "gamma": snapshot.get("gamma"),
                "theta": snapshot.get("theta"),
                "vega": snapshot.get("vega"),
            }
            for snapshot in raw_snapshots
        ]

    def get_iv_percentile(self, symbol: str, ts: Optional[datetime] = None) -> list[dict]:
        raw = self._get_iv_percentile_raw(symbol)
        row_ts = ts or datetime.now(timezone.utc)
        return [
            {"symbol": symbol, "ts": row_ts, "period": period, "percentile": raw.get(period)}
            for period in ("13w", "26w", "52w")
        ]
