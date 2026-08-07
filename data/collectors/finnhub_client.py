"""Finnhub 免费层 API 客户端——跟 `ibkr_client.py` 不同，这里写的是真实的
HTTP 请求代码，不是 Mock（Finnhub 是公开的真实 API，跟"还没部署好、验证
不了"的 IBKR CPAPI 不是一回事）。

⚠️ 重要限制：以下端点路径和返回字段是按我训练时掌握的 Finnhub API 文档
写的，Finnhub 的 API 有可能在此之后调整过端点路径、字段名，或者把某些
字段挪到付费层。**上线前务必拿真实 API Key 跑一次，对照 Finnhub 官方
最新文档核对端点是否还在、字段名是否还一致**，不要直接当成已验证事实
使用——这跟本仓库其它地方"未经验证的假设需要标注"的原则是一致的，只是
这次的假设是"API 格式记得对不对"，不是"业务规则合不合理"。

本次任务不要求真的发一次网络请求验证（也没有真实 API Key 可用），所以
`FinnhubClient` 支持注入自定义 `transport` 函数替换默认的 `urllib`
实现，测试用这个机制喂假 JSON 响应，不会真的连网。
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable, Optional


class FinnhubClient:
    """Finnhub REST API 客户端。

    真实请求都走 `_get()`：拼 URL（自动带上 `token` 参数）-> 调用
    `transport(url)` -> 期望拿到已经解析好的 JSON（dict 或 list）。
    默认 `transport` 用 `urllib.request` 发起真实 HTTP GET；测试/开发
    可以传自定义 `transport` 替换掉，避免真的连网。
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        transport: Optional[Callable[[str], object]] = None,
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport or self._default_transport

    def _default_transport(self, url: str) -> object:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - 官方公开 API，非用户输入拼接
            raw = resp.read()
        return json.loads(raw) if raw else None

    def _get(self, path: str, params: dict) -> object:
        query = {k: v for k, v in params.items() if v is not None}
        query["token"] = self.api_key
        url = f"{self.BASE_URL}{path}?{urllib.parse.urlencode(query)}"
        return self._transport(url)

    def get_earnings_calendar(
        self, from_date: str, to_date: str, symbol: Optional[str] = None
    ) -> dict:
        """GET /calendar/earnings —— 财报日历 + 历史 EPS/营收 actual vs estimate。

        真实返回大致形如：
            {"earningsCalendar": [
                {"date": "2024-01-25", "epsActual": 2.18, "epsEstimate": 2.10,
                 "hour": "amc", "quarter": 1, "revenueActual": 119575000000,
                 "revenueEstimate": 117910000000, "symbol": "AAPL", "year": 2024},
                ...
            ]}

        `from_date`/`to_date` 格式 "YYYY-MM-DD"。不传 `symbol` 时返回该
        时间范围内所有标的的财报（Finnhub 免费层支持批量查询，不是只能
        按标的一个个查）。
        """
        return self._get("/calendar/earnings", {"from": from_date, "to": to_date, "symbol": symbol})

    def get_upgrade_downgrade(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> list:
        """GET /stock/upgrade-downgrade —— 分析师评级变动（含机构名称）。

        真实返回大致形如：
            [{"symbol": "AAPL", "gradeTime": 1611878400, "company": "Wedbush",
              "fromGrade": "Neutral", "toGrade": "Outperform", "action": "up"}, ...]

        ⚠️ 这个端点不带目标价（`price_target`）字段——Finnhub 的目标价是
        另一个端点 `/stock/price-target`，返回的是全市场分析师目标价的
        聚合统计（`targetMean`/`targetHigh`/`targetLow`/`targetMedian`），
        不是"这家机构给的目标价是多少"，跟 `analyst_ratings` 表按
        (symbol, ts, firm) 存单家机构一条记录的设计对不上——这是数据源和
        表结构之间的一个真实缺口，不是采集代码的问题，`analyst_ratings`
        采集函数里会把 `price_target` 留空，具体怎么补（要不要额外查
        price-target 端点、按什么规则分摊到单家机构）需要你确认。
        """
        return self._get(
            "/stock/upgrade-downgrade", {"symbol": symbol, "from": from_date, "to": to_date}
        )

    def get_social_sentiment(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> dict:
        """GET /stock/social-sentiment —— Reddit/Twitter 社交媒体情绪
        （规格书 3.2 节"社交媒体情绪打分，MSPR sentiment"对应的数据源）。

        真实返回大致形如：
            {"symbol": "AAPL",
             "reddit": [{"atTime": "2024-01-25 00:00:00", "mention": 10,
                         "positiveScore": 0.6, "negativeScore": 0.4,
                         "positiveMention": 6, "negativeMention": 4, "score": 0.2}],
             "twitter": [...]}

        `score` 字段就是 MSPR 口径的情绪分数，映射到 `social_sentiment`
        表的 `mspr` 列。

        ⚠️ 这个端点在 Finnhub 免费层是否仍然可用、字段名是否还是这样，
        属于本文件头部说明的"需要用真实 API Key 核对"的范围之一——社交
        情绪类端点是 Finnhub 历史上调整比较频繁的一块。
        """
        return self._get(
            "/stock/social-sentiment", {"symbol": symbol, "from": from_date, "to": to_date}
        )
