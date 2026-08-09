"""IBKR 数据采集：正股 OHLCV / 期权链+行情快照 / IV 百分位。

⚠️ **已废弃，不再是默认连接方案**：跟同目录 `client.py` 一样，这个文件
连同它解析的"CPAPI 原始返回格式"整体属于已经换掉的 IBeam 方案，只保留
供回退参考，不再是 `brokers.ibkr` 的默认导出。新代码走
`brokers/ibkr/client.py` 里的 `IBInsyncClient`。

以下是换方案之前的原始说明（未改动，仅供参考）：

⚠️ 结构重构说明：这个文件是从 `data/collectors/ibkr_collector.py`
原样搬过来的（本次是纯结构调整，不改变任何解析逻辑）。惟一的改动是：
`collect_option_chain_and_snapshot` 和 `collect_iv_percentile` 调用的
客户端方法名从 `client.get_option_chain(...)`/`client.get_iv_percentile(...)`
改成了 `client._get_option_chain_raw(...)`/`client._get_iv_percentile_raw(...)`
——因为 `IBKRClient` 现在要实现 `brokers.base.BrokerClient` 定义的统一接口，
而那个接口里也有名叫 `get_option_chain`/`get_iv_percentile` 的方法（返回
`data/schema.sql` 表结构的行），跟这里原来"CPAPI 原始返回格式"的同名方法
签名不同、返回格式也不同，两者不能同时用同一个方法名挂在 `IBKRClient`
上，所以把这两个「返回 CPAPI 原始格式」的方法改成了私有命名
（见 `client.py` 里的重命名说明）。除了这两处方法名调用之外，本文件
每个函数的解析逻辑（字段映射、None 处理规则等）逐行未变。

每个 `collect_*` 函数职责一致：调用 `IBKRClient`（或 `MockIBKRClient`）
拿到假想的 API 原始返回，再用对应的 `_parse_*` 函数把它转换成跟
`data/schema.sql` 表字段一一对应的 dict 列表——每个 dict 就是"待写入
数据库的一行"，key 名直接是表的列名，方便以后接一层简单的
`INSERT INTO ... VALUES (%(col)s, ...)` 就能用（本次任务不要求写这层）。

本文件不做任何网络请求判断、重试、限流之类的基础设施逻辑——那些属于
`IBKRClient` 真实实现该管的事，采集函数只管"调用客户端 -> 解析成表行"。
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # 只在静态类型检查时导入，避免和 client.py 互相 import 造成循环依赖
    # （client.py 需要在模块顶层导入这个文件里的 collect_* 函数，来实现
    # `brokers.base.BrokerClient` 定义的统一接口方法）。
    from .client import IBKRClient


def _epoch_ms_to_datetime(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)


def collect_stock_ohlcv(
    client: "IBKRClient",
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> list[dict]:
    """采集正股 OHLCV，对应 `stock_ohlcv` 表。

    Parameters
    ----------
    client : `IBKRClient`（真实或 Mock）。
    symbol : 标的代码。
    timeframe : K 线周期，直接存进 `stock_ohlcv.timeframe` 列。
    start, end : 转发给 `client.get_price_history`。

    Returns
    -------
    list[dict]，每个 dict 的 key 是 `stock_ohlcv` 的列名：
    symbol/ts/timeframe/open/high/low/close/volume。
    """
    raw_bars = client.get_price_history(symbol, timeframe, start, end)
    return [_parse_ohlcv_bar(symbol, timeframe, bar) for bar in raw_bars]


def _parse_ohlcv_bar(symbol: str, timeframe: str, bar: dict) -> dict:
    return {
        "symbol": symbol,
        "ts": _epoch_ms_to_datetime(bar["t"]),
        "timeframe": timeframe,
        "open": bar.get("o"),
        "high": bar.get("h"),
        "low": bar.get("l"),
        "close": bar.get("c"),
        "volume": bar.get("v"),
    }


def collect_option_chain_and_snapshot(
    client: "IBKRClient", underlying: str, expiration_date: date
) -> tuple[list[dict], list[dict]]:
    """采集某个到期日的期权链结构 + 行情快照，对应 `option_contracts` +
    `option_snapshot` 两张表。

    先拿链结构确定这个到期日有哪些合约（`option_contracts`），再拿这批
    合约 ID 的行情快照（`option_snapshot`）——两步之间有依赖关系（快照
    需要先知道合约 ID 是什么），所以放在同一个函数里按顺序调用，不拆成
    两个互相不知道对方的独立函数。

    Parameters
    ----------
    client : `IBKRClient`（真实或 Mock）。
    underlying : 标的代码。
    expiration_date : 到期日。

    Returns
    -------
    (contract_rows, snapshot_rows)：
    - contract_rows 每个 dict 的 key 是 `option_contracts` 的列名：
      contract_id/underlying/expiration_date/strike/right
      （`created_at` 列有 `DEFAULT now()`，不需要采集层填，写库时让数据库
      自己填）。
    - snapshot_rows 每个 dict 的 key 是 `option_snapshot` 的列名：
      contract_id/ts/last_price/bid/ask/volume/open_interest/implied_vol/
      delta/gamma/theta/vega。
    """
    raw_contracts = client._get_option_chain_raw(underlying, expiration_date)
    contract_rows = [_parse_option_contract(underlying, expiration_date, c) for c in raw_contracts]

    contract_ids = [row["contract_id"] for row in contract_rows]
    raw_snapshots = client.get_option_market_data(contract_ids)
    snapshot_ts = datetime.now(timezone.utc)
    snapshot_rows = [_parse_option_snapshot(s, snapshot_ts) for s in raw_snapshots]

    return contract_rows, snapshot_rows


def _parse_option_contract(underlying: str, expiration_date: date, contract: dict) -> dict:
    return {
        "contract_id": contract["conid"],
        "underlying": underlying,
        "expiration_date": expiration_date,
        "strike": contract.get("strike"),
        "right": contract.get("right"),
    }


def _parse_option_snapshot(snapshot: dict, ts: datetime) -> dict:
    return {
        "contract_id": snapshot["conid"],
        "ts": ts,
        "last_price": snapshot.get("last"),
        "bid": snapshot.get("bid"),
        "ask": snapshot.get("ask"),
        "volume": snapshot.get("volume"),
        "open_interest": snapshot.get("open_interest"),
        "implied_vol": snapshot.get("implied_vol"),
        # Greeks 缺失是正常情况——深度虚值/流动性差的合约，IBKR 有时不返回
        # Greeks 字段，缺了就是 None，不是 0（0 是"确实算出来是0"，跟"没
        # 返回"含义不同，交给 Engine 层后续用 IV 反推补上，见 schema.sql
        # 里 option_snapshot.delta 列的注释）。
        "delta": snapshot.get("delta"),
        "gamma": snapshot.get("gamma"),
        "theta": snapshot.get("theta"),
        "vega": snapshot.get("vega"),
    }


def collect_iv_percentile(client: "IBKRClient", symbol: str, ts: Optional[datetime] = None) -> list[dict]:
    """采集 IV 百分位（13/26/52 周三个周期），对应 `iv_percentile` 表。

    Parameters
    ----------
    client : `IBKRClient`（真实或 Mock）。
    symbol : 标的代码。
    ts : 采集时间戳，不提供则用当前时间（UTC）。

    Returns
    -------
    list[dict]，固定 3 行（每个周期一行），key 是 `iv_percentile` 的列名：
    symbol/ts/period/percentile。某个周期数据缺失时该行 `percentile` 为
    `None`，不会因为缺一个周期就漏掉另外两个周期的行。
    """
    raw = client._get_iv_percentile_raw(symbol)
    row_ts = ts or datetime.now(timezone.utc)

    return [
        {"symbol": symbol, "ts": row_ts, "period": period, "percentile": raw.get(period)}
        for period in ("13w", "26w", "52w")
    ]
