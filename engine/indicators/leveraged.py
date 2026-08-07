"""第四批 —— 杠杆 ETF 对冲指标：波动损耗率回归、VIX 期限结构比值。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecayRegressionResult:
    beta: float           # 回归斜率，理想情况下应接近目标杠杆倍数
    daily_decay: float     # 回归截距，即扣除杠杆敞口后的"每日损耗率"（预期为负）
    r_squared: float


def volatility_decay_regression(
    etf_returns: list[float] | np.ndarray,
    underlying_returns: list[float] | np.ndarray,
    leverage: float,
) -> DecayRegressionResult:
    """杠杆 ETF 波动损耗率回归。

    杠杆 ETF 每日再平衡机制下，其日收益率理论上满足：

        r_etf_t ≈ leverage * r_underlying_t + daily_decay

    其中 daily_decay 主要来自"波动损耗"（volatility decay / beta slippage）：
    标的高波动、横盘震荡时，杠杆 ETF 会持续跑输 `leverage * 标的累计收益`，
    daily_decay 理论值近似为 -0.5 * (leverage^2 - leverage) * σ^2（σ 为标的
    日收益率标准差），符号通常为负。

    本函数用最小二乘线性回归 r_etf ~ beta * r_underlying + alpha 拟合真实
    数据，beta 应接近传入的 `leverage`（用于验证跟踪是否偏离），alpha 即为
    实测的每日损耗率。

    Parameters
    ----------
    etf_returns : 杠杆 ETF 的日收益率序列（小数形式，例如 0.01 表示 1%）。
    underlying_returns : 标的资产（或对应 1x 指数）的日收益率序列，长度需与
        etf_returns 一致、按同一日期对齐。
    leverage : 该 ETF 的名义杠杆倍数（例如 2x ETF 传 2.0，-1x 反向 ETF 传 -1.0），
        仅用于结果对比参考，不参与拟合。

    Returns
    -------
    DecayRegressionResult(beta, daily_decay, r_squared)
    """
    y = np.asarray(etf_returns, dtype=float)
    x = np.asarray(underlying_returns, dtype=float)
    if len(x) != len(y):
        raise ValueError("etf_returns 与 underlying_returns 长度必须一致")
    if len(x) < 2:
        raise ValueError("至少需要 2 个观测点才能做回归")

    beta, alpha = np.polyfit(x, y, 1)

    y_pred = beta * x + alpha
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return DecayRegressionResult(beta=float(beta), daily_decay=float(alpha), r_squared=float(r_squared))


def vix_term_structure_ratio(front_vix: float, back_vix: float) -> dict:
    """VIX 期限结构比值（例如 VIX9D/VIX，或 VIX/VIX3M）。

    ratio = front_vix / back_vix

    - ratio < 1：正向期限结构（Contango），市场平静时的常态，近月恐慌情绪
      低于远月，杠杆 ETF/波动率产品的展期损耗（roll decay）通常为负；
    - ratio > 1：逆向期限结构（Backwardation），近月恐慌情绪高于远月，
      往往对应市场短期恐慌、波动加剧。

    Parameters
    ----------
    front_vix : 近月波动率指数读数（例如 VIX9D 或 VIX）。
    back_vix : 远月波动率指数读数（例如 VIX 或 VIX3M）。

    Returns
    -------
    dict，包含 ratio 与 structure（"contango" / "backwardation"）。
    """
    if back_vix == 0:
        raise ValueError("back_vix 不能为 0")
    ratio = front_vix / back_vix
    structure = "backwardation" if ratio > 1 else "contango"
    return {"ratio": float(ratio), "structure": structure}
