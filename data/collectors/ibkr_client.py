"""IBKR Client Portal Web API（CPAPI）客户端接口层。

`IBKRClient` 只定义方法签名，真实实现（直连 IBeam 网关后调用 CPAPI）目前
全部 `raise NotImplementedError`——README 里写的很清楚，这个仓库还在
"M0 基础设施与可行性验证"阶段，Gateway 认证自动化（IBeam + Auto-Restart）
和直连 CPAPI 复测都还没做完，真的把网络请求代码写在这里、此刻却验证不了，
反而容易埋下"看起来实现了但从没跑通过"的假象。等 IBeam 部署好、能实际
发请求验证字段名之后，再把 `NotImplementedError` 换成真实的 HTTP 调用。

`MockIBKRClient` 是给测试/开发用的假实现，返回构造好的固定假数据，不发
任何网络请求，方法签名跟 `IBKRClient` 完全一致，采集函数
（`ibkr_collector.py`）不关心传进来的是真实客户端还是 Mock，这就是"可
替换的客户端接口层"的意思。

⚠️ 已通过 IBKR MCP 连接器/官方文档验证过的 CPAPI 字段，目前只有 Delta
(7308)/Gamma(7309)/Theta(7310)/Vega(7311) 这四个（见规格书第 12 章）。
Last/Bid/Ask/Volume/Open Interest 等其它字段的具体 field code，需要直连
CPAPI 之后对照官方文档核实，本文件里的方法只按"语义"命名（`last`/
`bid`/`ask`/...），不在注释里编造具体 field code 数字，避免以后被当成
"已验证事实"误用。
"""
from __future__ import annotations

from datetime import date
from typing import Optional


class IBKRClient:
    """CPAPI 客户端接口，真实实现留空。

    方法签名先按"采集函数需要什么"设计好，等直连 Gateway 之后再逐个实现。
    """

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
        由 `ibkr_collector._parse_ohlcv_bar` 解析成 `stock_ohlcv` 表的行。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
        """某个到期日的期权链结构（行使价 + 合约 ID），对应 CPAPI
        `/iserver/secdef/strikes` + `/iserver/secdef/info` 两步调用组合
        （先拿行使价列表，再逐个行使价查 conid）。

        Returns
        -------
        list[dict]，每个 dict 至少含合约 ID、行使价、Call/Put 标记（真实
        字段名待验证），由 `ibkr_collector._parse_option_contract` 解析成
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
        `ibkr_collector._parse_option_snapshot` 解析成 `option_snapshot`
        表的行。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")

    def get_iv_percentile(self, symbol: str) -> dict:
        """IV 百分位（13/26/52 周），对应 CPAPI 现成字段
        `implied_volatility_percentile`（规格书 3.1 节已确认存在，具体
        怎么区分 13/26/52 周三个周期待直连后确认）。

        Returns
        -------
        dict，含三个周期的百分位数值，由
        `ibkr_collector._parse_iv_percentile` 解析成 `iv_percentile`
        表的三行（每个周期一行）。
        """
        raise NotImplementedError("直连 CPAPI 的实现待 IBeam 网关搭好后补上")


class MockIBKRClient(IBKRClient):
    """测试/开发用的假实现，返回构造好的固定假数据，不发任何网络请求。

    默认数据是硬编码的固定值（不是随机生成），方便测试里手算核对解析结果；
    也可以在构造时传 `fixture` 覆盖，测试"缺字段"等边界情况。
    """

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

    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
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

    def get_iv_percentile(self, symbol: str) -> dict:
        if "iv_percentile" in self._fixture:
            return self._fixture["iv_percentile"]
        return {"13w": 45.2, "26w": 52.8, "52w": 61.5}
