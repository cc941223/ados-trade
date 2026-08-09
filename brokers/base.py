"""券商数据读取层的统一接口。

这是一次纯结构重构抽出来的抽象基类：在此之前，`engine/pipeline` 的四个
场景编排函数从来没有直接依赖过 `data/collectors` 里任何具体的券商实现
（`IBKRClient`/`MockIBKRClient`）——它们只接受已经取好的 `pandas`
DataFrame/浮点数作为参数，是纯计算函数。这个基类现在存在的意义是：
把"读取行情"这个动作的方法签名固定下来，往后如果编排层、采集脚本或者
别的券商实现要依赖"一个提供行情读取能力的对象"，依赖的是这里声明的
接口，不是某个具体券商 SDK 的方法名。

四个方法名和返回的行结构对照 `data/schema.sql` 里 IBKR 数据源相关的表：

- `get_stock_ohlcv`      -> `stock_ohlcv` 表
- `get_option_chain`     -> `option_contracts` 表
- `get_option_snapshot`  -> `option_snapshot` 表
- `get_iv_percentile`    -> `iv_percentile` 表

也就是说，这个接口的方法直接返回"跟数据库表字段一一对应的 dict 列表"
（`list[dict]`），而不是某个券商 API 的原始返回格式——原始 API 格式属于
具体实现（例如 IBKR CPAPI 的字段名）的内部细节，不应该出现在这层抽象里。

⚠️ 本次只做数据读取层的抽象，不包含 `place_order` 之类的下单方法——那是
另一个量级的抽象（涉及订单状态机、幂等、审计等），本次重构范围之外，
故意留白，不是遗漏。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional


class BrokerClient(ABC):
    """券商数据读取层的统一接口，只负责"读"，不负责"写"（不下单）。"""

    @abstractmethod
    def get_stock_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict]:
        """正股 OHLCV 历史数据，对应 `stock_ohlcv` 表。

        Returns
        -------
        list[dict]，每个 dict 的 key 是 `stock_ohlcv` 的列名：
        symbol/ts/timeframe/open/high/low/close/volume。
        """
        raise NotImplementedError

    @abstractmethod
    def get_option_chain(self, underlying: str, expiration_date: date) -> list[dict]:
        """某个到期日的期权链结构，对应 `option_contracts` 表。

        Returns
        -------
        list[dict]，每个 dict 的 key 是 `option_contracts` 的列名：
        contract_id/underlying/expiration_date/strike/right。
        """
        raise NotImplementedError

    @abstractmethod
    def get_option_snapshot(self, contract_ids: list[int]) -> list[dict]:
        """一批期权合约的行情快照（含 IV/Greeks），对应 `option_snapshot` 表。

        跟 `get_option_chain` 是两个独立方法而不是一个合并调用：调用方拿到
        `get_option_chain` 返回的 `contract_id` 之后自己决定要给哪些合约
        查快照，接口不替调用方做这个决定（对照 `option_contracts` 和
        `option_snapshot` 本来就是 schema.sql 里两张独立的表，只共享
        `contract_id` 这一个外键）。

        Parameters
        ----------
        contract_ids : 要查询快照的合约 ID 列表（`option_contracts.contract_id`）。

        Returns
        -------
        list[dict]，每个 dict 的 key 是 `option_snapshot` 的列名：
        contract_id/ts/last_price/bid/ask/volume/open_interest/implied_vol/
        delta/gamma/theta/vega。Greeks 缺失时对应字段是 `None`，不是 `0`
        （`0` 表示"确实算出来是 0"，跟"没返回"含义不同）。
        """
        raise NotImplementedError

    @abstractmethod
    def get_iv_percentile(self, symbol: str, ts: Optional[datetime] = None) -> list[dict]:
        """IV 百分位（13/26/52 周），对应 `iv_percentile` 表。

        Parameters
        ----------
        symbol : 标的代码。
        ts : 采集时间戳，不提供则由实现自己决定（通常是当前时间 UTC）。

        Returns
        -------
        list[dict]，固定 3 行（每个周期一行），key 是 `iv_percentile` 的
        列名：symbol/ts/period/percentile。某个周期数据缺失时该行
        `percentile` 为 `None`，不会因为缺一个周期就漏掉另外两个周期的行。
        """
        raise NotImplementedError
