"""brokers/ibkr/collector.py 单元测试 —— 用 MockIBKRClient 构造假数据，
验证解析结果跟 schema.sql 字段对得上，覆盖正常情况和缺字段的情况。

⚠️ 结构重构说明：这个文件是从 `data/collectors/tests/test_ibkr_collector.py`
原样搬过来的，惟一改动是 import 路径（`data.collectors.ibkr_client`/
`data.collectors.ibkr_collector` -> `brokers.mock`/`brokers.ibkr.collector`），
断言内容逐条未变。
"""
from datetime import date, datetime, timezone

from brokers.ibkr.collector import (
    collect_iv_percentile,
    collect_option_chain_and_snapshot,
    collect_stock_ohlcv,
)
from brokers.mock import MockIBKRClient


# ---------------------------------------------------------------------------
# stock_ohlcv
# ---------------------------------------------------------------------------


def test_collect_stock_ohlcv_normal_case():
    client = MockIBKRClient(
        fixture={
            "price_history": [
                {"t": 1735689600000, "o": 190.0, "h": 192.5, "l": 189.0, "c": 191.2, "v": 45_000_000},
                {"t": 1735776000000, "o": 191.5, "h": 193.0, "l": 190.5, "c": 192.8, "v": 38_000_000},
            ]
        }
    )

    rows = collect_stock_ohlcv(client, symbol="AAPL", timeframe="1day")

    assert len(rows) == 2
    assert set(rows[0]) == {"symbol", "ts", "timeframe", "open", "high", "low", "close", "volume"}

    first = rows[0]
    assert first["symbol"] == "AAPL"
    assert first["timeframe"] == "1day"
    # 1735689600000 ms = 2025-01-01T00:00:00Z
    assert first["ts"] == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert first["open"] == 190.0
    assert first["high"] == 192.5
    assert first["low"] == 189.0
    assert first["close"] == 191.2
    assert first["volume"] == 45_000_000


def test_collect_stock_ohlcv_missing_optional_price_field():
    # 某根K线缺 low（数据源偶发缺字段），不应该整条采集报错
    client = MockIBKRClient(
        fixture={"price_history": [{"t": 1735689600000, "o": 190.0, "h": 192.5, "c": 191.2, "v": 45_000_000}]}
    )

    rows = collect_stock_ohlcv(client, symbol="AAPL", timeframe="1day")

    assert rows[0]["low"] is None
    assert rows[0]["open"] == 190.0


def test_collect_stock_ohlcv_uses_default_fixture_without_override():
    client = MockIBKRClient()
    rows = collect_stock_ohlcv(client, symbol="TSLA", timeframe="5min")
    assert len(rows) == 2
    assert all(row["symbol"] == "TSLA" and row["timeframe"] == "5min" for row in rows)


# ---------------------------------------------------------------------------
# option_contracts + option_snapshot
# ---------------------------------------------------------------------------


def test_collect_option_chain_and_snapshot_normal_case():
    client = MockIBKRClient()

    contract_rows, snapshot_rows = collect_option_chain_and_snapshot(
        client, underlying="AAPL", expiration_date=date(2026, 9, 18)
    )

    assert len(contract_rows) == 4
    assert set(contract_rows[0]) == {"contract_id", "underlying", "expiration_date", "strike", "right"}

    first_contract = contract_rows[0]
    assert first_contract["contract_id"] == 500001
    assert first_contract["underlying"] == "AAPL"
    assert first_contract["expiration_date"] == date(2026, 9, 18)
    assert first_contract["strike"] == 190.0
    assert first_contract["right"] == "C"

    assert len(snapshot_rows) == 4
    assert set(snapshot_rows[0]) == {
        "contract_id", "ts", "last_price", "bid", "ask", "volume", "open_interest",
        "implied_vol", "delta", "gamma", "theta", "vega",
    }

    first_snapshot = next(s for s in snapshot_rows if s["contract_id"] == 500001)
    assert first_snapshot["last_price"] == 5.20
    assert first_snapshot["bid"] == 5.10
    assert first_snapshot["ask"] == 5.30
    assert first_snapshot["delta"] == 0.62
    assert isinstance(first_snapshot["ts"], datetime)


def test_collect_option_snapshot_missing_greeks_becomes_none_not_zero():
    # 深度虚值合约没有 Greeks 字段（IBKR 偶发不返回），应该是 None 不是 0
    client = MockIBKRClient(
        fixture={
            "option_chain": [{"conid": 999001, "strike": 300.0, "right": "C"}],
            "option_market_data": [
                {"conid": 999001, "last": 0.05, "bid": 0.01, "ask": 0.10, "volume": 5, "open_interest": 20}
            ],
        }
    )

    contract_rows, snapshot_rows = collect_option_chain_and_snapshot(
        client, underlying="AAPL", expiration_date=date(2026, 9, 18)
    )

    assert len(contract_rows) == 1
    assert snapshot_rows[0]["delta"] is None
    assert snapshot_rows[0]["gamma"] is None
    assert snapshot_rows[0]["implied_vol"] is None
    # 有值的字段不受影响
    assert snapshot_rows[0]["last_price"] == 0.05


def test_collect_option_chain_empty_case_returns_empty_snapshot_too():
    client = MockIBKRClient(fixture={"option_chain": []})

    contract_rows, snapshot_rows = collect_option_chain_and_snapshot(
        client, underlying="AAPL", expiration_date=date(2026, 9, 18)
    )

    assert contract_rows == []
    assert snapshot_rows == []


# ---------------------------------------------------------------------------
# iv_percentile
# ---------------------------------------------------------------------------


def test_collect_iv_percentile_normal_case():
    client = MockIBKRClient()
    fixed_ts = datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc)

    rows = collect_iv_percentile(client, symbol="AAPL", ts=fixed_ts)

    assert len(rows) == 3
    assert {row["period"] for row in rows} == {"13w", "26w", "52w"}
    assert all(row["symbol"] == "AAPL" and row["ts"] == fixed_ts for row in rows)

    by_period = {row["period"]: row["percentile"] for row in rows}
    assert by_period["13w"] == 45.2
    assert by_period["26w"] == 52.8
    assert by_period["52w"] == 61.5


def test_collect_iv_percentile_missing_one_period_keeps_other_two():
    client = MockIBKRClient(fixture={"iv_percentile": {"13w": 40.0, "52w": 55.0}})  # 缺 26w

    rows = collect_iv_percentile(client, symbol="AAPL")

    by_period = {row["period"]: row["percentile"] for row in rows}
    assert by_period["13w"] == 40.0
    assert by_period["26w"] is None
    assert by_period["52w"] == 55.0


def test_collect_iv_percentile_default_ts_is_generated():
    client = MockIBKRClient()
    rows = collect_iv_percentile(client, symbol="AAPL")
    assert all(isinstance(row["ts"], datetime) for row in rows)
