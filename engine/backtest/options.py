"""6.2 节 —— 期权策略场景成功判定：evaluate_options_signal()。

退出规则：达到最大理论盈利 50% 时提前平仓，或剩余 21 天到期时强制平仓，
两者先触发者为准；需单独统计持仓期间最大浮亏。

6.3 节硬性要求：期权回测使用买一/卖一价模拟真实成交，不用中间价——本函数
按 `position_side` 决定用 `bid`（多头平仓，卖出吃买一价）还是 `ask`
（空头平仓，买回吃卖一价），从头到尾没有用过 mid price。
"""
from __future__ import annotations

import pandas as pd

from .base import SignalOutcome


def evaluate_options_signal(
    entry_price: float,
    price_path: pd.DataFrame,
    position_side: str = "long",
    profit_target_pct: float = 0.5,
    time_exit_dte: int = 21,
) -> SignalOutcome:
    """判定一笔期权信号的胜负，逐日检查"50% 盈利"和"剩余 21 天到期"两条
    退出规则，谁先触发就按谁退出。

    Parameters
    ----------
    entry_price : 入场时的期权权利金——`position_side="long"` 时是买入
        付出的权利金（应该来自入场那一刻的 `ask`，因为买入吃卖一价），
        `position_side="short"` 时是卖出收到的权利金（应该来自入场那一刻
        的 `bid`，因为卖出吃买一价）。这个价格由调用方在入场时就按方向
        选对了传进来，本函数不重复做这一步。
    price_path : 入场后逐日的期权报价，需包含 `bid`/`ask`/`dte`
        （剩余到期天数）三列，按时间升序排列。**不能包含中间价列参与
        任何计算**（6.3 节硬性要求），本函数只读 `bid`/`ask`。
    position_side : "long"（多头，平仓时卖出，吃 `bid`）或 "short"
        （空头/卖方，例如 covered call 的空头 call 那一腿，平仓时买回，
        吃 `ask`）。
    profit_target_pct : 提前平仓的盈利目标，默认 0.5（50%，对应规格书
        原文"最大理论盈利 50%"）。

        ⚠️ V1 占位值，待 M3 回测校准：规格书写的是具体数字"50%"，但按
        7.3 节对同类系数的说明口径（这类具体数字本身也是主观设定，需要
        回测验证是否真的是最优退出点），性质跟本项目其它 V1 占位系数一致。
    time_exit_dte : 剩余到期天数达到此值时强制平仓，默认 21（规格书原文
        "剩余 21 天到期"），同样是待 M3 校准的占位值。

    Returns
    -------
    SignalOutcome：
    - 盈利目标先触发 -> outcome="win", exit_reason="target_hit"
    - 到期日窗口先触发（DTE 降到 `time_exit_dte`）-> outcome 按当时盈亏
      正负判定（"loss"其实是负例，参照两个都提前触发；平仓仍有正盈亏
      即"win"，非负值都判 win，只有严格 <0 才判 loss——规格书原文没有
      明确给出"到期强制平仓但仍小赚"算不算成功，这是我的补充规则：
      退出规则只回答"什么时候退出"，没回答"退出后怎么判定输赢"，
      我的处理是按最终盈亏的正负号判定，跟其它三个场景"看最终收益是否
      为正"的判定精神一致，需要你确认），exit_reason="time_exit"
    - 数据在两条件都触发前就结束（比如价格数据没接到剩 21 天那一刻）
      -> outcome="undecided", exit_reason="undecided"

    `max_drawdown_pct` 是持仓期内（含最终退出那一天）观察到的最差浮亏，
    单独统计，跟最终 `outcome`/`raw_return_pct` 分开记录，符合规格书
    "需单独统计持仓期间最大浮亏，不能仅看最终盈亏"的要求。
    """
    if position_side not in ("long", "short"):
        raise ValueError(f"position_side 必须是 'long' 或 'short'，收到：{position_side!r}")

    max_drawdown = 0.0
    last_ts = None
    last_pnl_pct = None
    last_exit_price = None

    for ts, row in price_path.iterrows():
        current_value = row["bid"] if position_side == "long" else row["ask"]
        if position_side == "long":
            pnl_pct = (current_value - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_value) / entry_price

        max_drawdown = min(max_drawdown, pnl_pct)
        last_ts, last_pnl_pct, last_exit_price = ts, pnl_pct, current_value

        target_hit = pnl_pct >= profit_target_pct
        time_exit = row["dte"] <= time_exit_dte

        if target_hit:
            # 盈利目标和到期窗口在同一天都成立时，按盈利目标处理——理由见
            # 函数文档。
            return SignalOutcome(
                outcome="win",
                exit_reason="target_hit",
                exit_price=current_value,
                exit_ts=ts,
                raw_return_pct=pnl_pct,
                max_drawdown_pct=max_drawdown,
            )
        if time_exit:
            outcome = "win" if pnl_pct >= 0 else "loss"
            return SignalOutcome(
                outcome=outcome,
                exit_reason="time_exit",
                exit_price=current_value,
                exit_ts=ts,
                raw_return_pct=pnl_pct,
                max_drawdown_pct=max_drawdown,
            )

    # 数据还没走到任何一个退出条件：未分出胜负，但仍然把已经观察到的浮盈亏
    # 和最大浮亏记下来，方便留痕。
    return SignalOutcome(
        outcome="undecided",
        exit_reason="undecided",
        exit_price=last_exit_price,
        exit_ts=last_ts,
        raw_return_pct=last_pnl_pct,
        max_drawdown_pct=max_drawdown,
    )
