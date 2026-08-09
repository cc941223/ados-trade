"""IBKR Client Portal Web API（CPAPI）客户端接口层。

`IBKRClient` 只定义方法签名，真实实现（直连 IBeam 网关后调用 CPAPI）目前
全部 `raise NotImplementedError`——README 里写的很清楚，这个仓库还在
"M0 基础设施与可行性验证"阶段，Gateway 认证自动化（IBeam + Auto-Restart）
和直连 CPAPI 复测都还没做完，真的把网络请求代码写在这里、此刻却验证不了，
反而容易埋下"看起来实现了但从没跑通过"的假象。等 IBeam 部署好、能实际
发请求验证字段名之后，再把 `NotImplementedError` 换成真实的 HTTP 调用。

⚠️ 已通过 IBKR MCP 连接器/官方文档验证过的 CPAPI 字段，目前只有 Delta
(7308)/Gamma(7309)/Theta(7310)/Vega(7311) 这四个（见规格书第 12 章）。
Last/Bid/Ask/Volume/Open Interest 等其它字段的具体 field code，需要直连
CPAPI 之后对照官方文档核实，本文件里的方法只按"语义"命名（`last`/
`bid`/`ask`/...），不在注释里编造具体 field code 数字，避免以后被当成
"已验证事实"误用。

⚠️ 结构重构说明（本次从 `data/collectors/ibkr_client.py` 原样搬过来）：

- 这个文件原来同时定义 `IBKRClient` 和 `MockIBKRClient`；`MockIBKRClient`
  现在搬到了 `brokers/mock/client.py`（对应 `brokers/mock/` 独立目录），
  这里只保留 `IBKRClient`。
- 新增了 `IBKRClient` 对 `brokers.base.BrokerClient` 统一接口的实现
  （`get_stock_ohlcv`/`get_option_chain`/`get_option_snapshot`/
  `get_iv_percentile` 四个方法），内部直接委托给 `collector.py` 里原有
  的 `collect_*` 解析函数，解析逻辑本身逐行未变。
- 原来两个返回"CPAPI 原始格式"的方法改了名字，加了下划线前缀：
  `get_option_chain` -> `_get_option_chain_raw`，
  `get_iv_percentile` -> `_get_iv_percentile_raw`。
  这是因为 `BrokerClient` 接口里也有同名的 `get_option_chain`/
  `get_iv_percentile`，但那两个方法返回的是 `data/schema.sql` 表结构的行
  （schema 形状），跟这里"CPAPI 原始返回格式"（`conid`/`strike`/`right`
  这种字段名）完全不是一回事，一个类不能用同一个名字挂两个不同签名/
  语义的方法，所以把原始格式的那两个改成私有命名腾出位置。
  `get_price_history`/`get_option_market_data` 这两个原始方法名跟接口
  没有冲突，维持原名不变。全文搜索确认过，除了这个文件和
  `collector.py` 内部，没有任何其它代码直接按名字调用这四个原始方法，
  改名不影响任何已有测试。
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from ..base import BrokerClient
from .collector import (
    _parse_option_contract,
    _parse_option_snapshot,
    collect_iv_percentile,
    collect_stock_ohlcv,
)


class IBKRClient(BrokerClient):
    """CPAPI 客户端接口，真实实现留空。

    方法签名先按"采集函数需要什么"设计好，等直连 Gateway 之后再逐个实现。
    """

    # ------------------------------------------------------------------
    # CPAPI 原始格式方法（真实实现待补，目前全部 raise NotImplementedError）
    # ------------------------------------------------------------------

    def get_price_history(
        self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[dict]:
        """正股 OHLCV 历史数据，对应 CPAPI `/iserver/marketdata/history`
        （或历史数据用 `/hmds/history`，具体用哪个待直连后确认）。

        Parameters
        ----------
        symbol : 标的代码。
        timeframe : K 线周期，例如 "1min"/"5min"/"1day"，跟
            `stock_ohlcv.timeframe` 列的取值约定一致。
        start, end : 时间范围（可选，具体格式待接入时按 CPAPI 实际要求调整）。

        Returns
        -------
        list[dict]，每个 dict 是一根 K 线的原始返回（真实字段名待验证），
        由 `collector._parse_ohlcv_bar` 解析成 `stock_ohlcv` 表的行。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    def _get_option_chain_raw(self, underlying: str, expiration_date: date) -> list[dict]:
        """某个到期日的期权链结构（行使价 + 合约 ID），对应 CPAPI
        `/iserver/secdef/strikes` + `/iserver/secdef/info` 两步调用组合
        （先拿行使价列表，再逐个行使价查 conid）。

        Returns
        -------
        list[dict]，每个 dict 至少含合约 ID、行使价、Call/Put 标记（真实
        字段名待验证），由 `collector._parse_option_contract` 解析成
        `option_contracts` 表的行。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    def get_option_market_data(self, contract_ids: list[int]) -> list[dict]:
        """一批期权合约的行情快照（含 IV/Greeks），对应 CPAPI
        `/iserver/marketdata/snapshot`。

        Returns
        -------
        list[dict]，每个 dict 对应一个合约的快照（真实字段名待验证，
        Delta/Gamma/Theta/Vega 已确认是 field code 7308-7311），由
        `collector._parse_option_snapshot` 解析成 `option_snapshot`
        表的行。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    def _get_iv_percentile_raw(self, symbol: str) -> dict:
        """IV 百分位（13/26/52 周），对应 CPAPI 现成字段
        `implied_volatility_percentile`（规格书 3.1 节已确认存在，具体
        怎么区分 13/26/52 周三个周期待直连后确认）。

        Returns
        -------
        dict，含三个周期的百分位数值，由
        `collector._parse_iv_percentile` 解析成 `iv_percentile`
        表的三行（每个周期一行）。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    # ------------------------------------------------------------------
    # brokers.base.BrokerClient 统一接口实现——委托给 collector.py 里的
    # collect_* 解析函数，本身不包含任何新的解析逻辑。
    # ------------------------------------------------------------------

    def get_stock_ohlcv(
        self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[dict]:
        return collect_stock_ohlcv(self, symbol, timeframe, start, end)

    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
        raw_contracts = self._get_option_chain_raw(underlying, expiration_date)
        return [_parse_option_contract(underlying, expiration_date, c) for c in raw_contracts]

    def get_option_snapshot(self, contract_ids: list[int]) -> list[dict]:
        raw_snapshots = self.get_option_market_data(contract_ids)
        snapshot_ts = datetime.now(timezone.utc)
        return [_parse_option_snapshot(s, snapshot_ts) for s in raw_snapshots]

    def get_iv_percentile(self, symbol: str, ts: Optional[datetime] = None) -> list[dict]:
        return collect_iv_percentile(self, symbol, ts)
