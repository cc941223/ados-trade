"""6.2 节 —— 波段 / 趋势跟踪场景成功判定：evaluate_swing_signal()。

持仓周期固定为信号触发后 N 个交易日（默认 10）。
超额收益 = 持仓期绝对收益 － 同期基准涨跌幅；超额收益 ＞ 0 才算成功。
"""
from __future__ import annotations

from .base import SignalOutcome


def evaluate_swing_signal(
    entry_price: float,
    exit_price: float,
    benchmark_entry_price: float,
    benchmark_exit_price: float,
    direction: str,
) -> SignalOutcome:
    """判定一笔波段信号在固定持仓周期（N 个交易日，由调用方决定 exit_price
    取自哪一天，本函数不做日期推算）结束时的超额收益胜负。

    Parameters
    ----------
    entry_price, exit_price : 标的自身在信号触发时、持仓期结束时的价格。
        `exit_price` 对应哪一天（入场后第几个交易日）由调用方决定——规格书
        6.2 节"N 初始取 10"是"该固定持仓多少天"的业务参数，跟具体某一次
        判定用的是哪两个价格点是分开的两件事：本函数只管"给定入场/退出两个
        价格点，超额收益是多少"，不管 N 具体是多少、退出日期怎么定位到
        K 线数据里，那部分是编排层（`engine.pipeline`）该做的数据整理，
        不是判定逻辑本身。
    benchmark_entry_price, benchmark_exit_price : 基准（如 QQQ 或对应板块
        ETF）在同一对时间点的价格。
    direction : "bull" 或 "bear"，决定收益率的符号——"bear" 方向的信号，
        价格下跌才是正收益（对应做空或反向头寸的盈亏方向）。

        ⚠️ 规格书原文"超额收益 = 持仓期绝对收益 － 同期基准涨跌幅"字面上
        没有提到方向调整，但波段场景的信号本来就有 Bull/Bear 两个方向
        （`score_swing` 输出 bull_score/bear_score），如果不按方向调整
        符号，bear 方向信号在价格下跌时会被算成"负收益"，这跟"方向判断
        对了应该算赢"的直觉矛盾。这里的处理是：先把标的自身收益率和基准
        收益率都按 `direction` 转成"这个方向上赚了多少"，再相减——
        等价于"如果你在基准上也做了同方向的头寸，你比基准好多少"。这是
        我的设计选择，规格书没有明确到这个细节，需要你确认是否合理。

    Returns
    -------
    SignalOutcome，`outcome` 为 "win"/"loss"（波段场景没有 undecided 状态
    ——固定持仓 N 天后一定会有退出价，不存在"还没分出胜负"的情况）。
    `raw_return_pct` 是方向调整后的自身收益率，`excess_return_pct` 是
    扣除基准后的超额收益。
    """
    if direction not in ("bull", "bear"):
        raise ValueError(f"direction 必须是 'bull' 或 'bear'，收到：{direction!r}")

    sign = 1.0 if direction == "bull" else -1.0

    asset_return = (exit_price - entry_price) / entry_price
    benchmark_return = (benchmark_exit_price - benchmark_entry_price) / benchmark_entry_price

    raw_return_pct = sign * asset_return
    excess_return_pct = raw_return_pct - sign * benchmark_return

    outcome = "win" if excess_return_pct > 0 else "loss"

    return SignalOutcome(
        outcome=outcome,
        exit_reason="time_exit",
        exit_price=exit_price,
        raw_return_pct=raw_return_pct,
        excess_return_pct=excess_return_pct,
    )
