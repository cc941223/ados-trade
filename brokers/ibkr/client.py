"""IBKR 真实数据读取实现：ib_insync + IB Gateway（Docker，TWS API 协议）。

M0 阶段的连接方式从"IBeam + Client Portal Web API（CPAPI，HTTPS）"换成了
"ib_insync + IB Gateway Docker（TWS API，socket 协议，端口 4002）"——旧的
IBeam/CPAPI 实现完整保留在 `brokers/ibkr/legacy_ibeam/`，仅供回退参考，
不再是默认路径。`docker-compose.yml` 现在启动的是 `ibgateway` 服务
（`ghcr.io/gnzsnz/ib-gateway`），对外暴露 4002（TWS API）和 5900（VNC，
首次登录/2FA 需要通过 VNC 桌面手动确认）。

`IBInsyncClient` 目前只实现了 `get_stock_ohlcv` 一个方法——先把这一个方法
跑通、确认连通性没问题，剩下三个方法（`get_option_chain`/
`get_option_snapshot`/`get_iv_percentile`）留着 `NotImplementedError`，
等 `get_stock_ohlcv` 验证过再逐个补上，不是遗漏。

⚠️ `get_stock_ohlcv` 里以下几处是本次"先跑通"阶段的占位设计，不是
已经验证过的最终方案，后续实现其它方法或者接入真实调度时应该重新核对：

- 合约类型硬编码成 `Stock(symbol, "SMART", "USD")`（美股 + SMART 路由 +
  美元结算）。规格书当前场景全部是美股，暂时够用，但如果以后要支持其它
  市场/币种，这里需要参数化。
- `whatToShow="TRADES"`、`useRTH=True`（只要常规交易时段）是 IB
  历史数据接口里最常见的默认组合，没有找到规格书对这两个参数的具体要求，
  先按最常见用法写，欢迎确认。
- `start`/`end` 参数目前只接受 `"YYYY-MM-DD"` 格式的日期字符串，内部转换
  成 IB 历史数据接口需要的 `durationStr`（"N D"）+ `endDateTime`。两者都
  不传时默认拉最近 30 天（`_DEFAULT_DURATION_DAYS`），这个默认值是没有
  在规格书或任务里明确要求的占位选择，不代表这是"正确"的默认回溯窗口。
- 没有做重试/断线重连——`ib_insync` 断线后 `reqHistoricalData` 会直接
  抛异常，调用方目前需要自己处理。
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from ib_insync import IB, Stock

from ..base import BrokerClient

# stock_ohlcv.timeframe 的取值约定（跟 schema.sql 注释一致）到 ib_insync
# `reqHistoricalData(barSizeSetting=...)` 需要的字符串之间的映射。
_TIMEFRAME_TO_BAR_SIZE = {
    "1min": "1 min",
    "5min": "5 mins",
    "15min": "15 mins",
    "30min": "30 mins",
    "1h": "1 hour",
    "1day": "1 day",
    "1d": "1 day",
}

# ⚠️ 占位默认值：start/end 都不传时的默认回溯天数，未在规格书里明确要求。
_DEFAULT_DURATION_DAYS = 30


class IBInsyncClient(BrokerClient):
    """通过 ib_insync 连接 IB Gateway（TWS API，默认 4002 端口）读取行情。

    Parameters
    ----------
    host : IB Gateway 地址，默认 "127.0.0.1"（docker-compose 把 4002 映射
        到宿主机同名端口，本机直连即可；如果调用方本身也跑在同一个
        docker network 里，可以传服务名 "ibgateway"）。
    port : TWS API 端口，默认 4002（`docker-compose.yml` 里 `ibgateway`
        服务映射的端口；4002 是 IB Gateway 的 paper 账户默认端口，实盘
        账户默认是 4001——本项目 M0 阶段固定连 paper，见
        `docker-compose.yml` 里 `TRADING_MODE: paper`）。
    client_id : TWS API 允许同一个 Gateway 被多个客户端同时连接，用
        `client_id` 区分，默认 1。如果同时跑多个连接（例如这个客户端
        再加一个手动检查脚本），必须用不同的 `client_id`，否则后连的会
        把先连的踢掉。
    timeout : 连接超时秒数，默认 10。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        self._ib: Optional[IB] = None

    def connect(self) -> None:
        """建立到 IB Gateway 的连接（幂等，已连接时不会重复连接）。"""
        if self._ib is not None and self._ib.isConnected():
            return
        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id, timeout=self._timeout)

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def get_stock_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict]:
        """正股 OHLCV 历史数据，对应 `stock_ohlcv` 表。

        Parameters
        ----------
        symbol : 标的代码（例如 "AAPL"）。
        timeframe : K 线周期，取值见 `_TIMEFRAME_TO_BAR_SIZE`。
        start, end : 可选，`"YYYY-MM-DD"` 格式；都不传则拉最近
            `_DEFAULT_DURATION_DAYS` 天到现在。

        Returns
        -------
        list[dict]，每个 dict 的 key 是 `stock_ohlcv` 的列名：
        symbol/ts/timeframe/open/high/low/close/volume。
        """
        bar_size = _TIMEFRAME_TO_BAR_SIZE.get(timeframe)
        if bar_size is None:
            raise ValueError(
                f"不支持的 timeframe: {timeframe!r}，目前支持 {sorted(_TIMEFRAME_TO_BAR_SIZE)}"
            )

        self.connect()

        start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None
        duration_str = self._duration_str(start_dt, end_dt)
        end_datetime = end_dt.strftime("%Y%m%d %H:%M:%S") if end_dt else ""

        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
        )

        return [
            {
                "symbol": symbol,
                "ts": bar.date if isinstance(bar.date, datetime) else datetime.combine(bar.date, datetime.min.time()),
                "timeframe": timeframe,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

    @staticmethod
    def _duration_str(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> str:
        if start_dt is None:
            return f"{_DEFAULT_DURATION_DAYS} D"
        reference_end = end_dt or datetime.utcnow()
        days = max(1, (reference_end - start_dt).days)
        return f"{days} D"

    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
        raise NotImplementedError("get_stock_ohlcv 先跑通，其它方法等确认连通性后再补")

    def get_option_snapshot(self, contract_ids: list[int]) -> list[dict]:
        raise NotImplementedError("get_stock_ohlcv 先跑通，其它方法等确认连通性后再补")

    def get_iv_percentile(self, symbol: str, ts: Optional[datetime] = None) -> list[dict]:
        raise NotImplementedError("get_stock_ohlcv 先跑通，其它方法等确认连通性后再补")
