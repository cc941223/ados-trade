"""期权相关指标。

第一批：Max Pain、Put/Call Ratio。
第三批：Black-Scholes 期权定价与 Greeks（Delta/Gamma/Theta/Vega）、隐含波动率反推、
        IV Skew、IV Term Structure。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 第一批：Max Pain / Put-Call Ratio
# ---------------------------------------------------------------------------


def max_pain(strikes: list[float], call_oi: list[float], put_oi: list[float]) -> float:
    """计算 Max Pain（最大痛点）。

    对每一个候选到期结算价 `K_i`（取自 strikes 列表），计算若结算价为
    `K_i` 时，所有期权买方合计能拿到的价值（即期权做市商/卖方需要支付
    的总额）：

        payout(K_i) = Σ_j max(K_i - strikes_j, 0) * call_oi_j
                    + Σ_j max(strikes_j - K_i, 0) * put_oi_j

    Max Pain 即使 payout 最小的那个结算价 —— 传统理论认为价格倾向于
    向该点靠近（对期权卖方最有利）。

    Parameters
    ----------
    strikes : 行使价列表。
    call_oi : 对应行使价的 Call 未平仓合约量（Open Interest）。
    put_oi : 对应行使价的 Put 未平仓合约量。

    Returns
    -------
    payout 最小对应的行使价（若有多个并列最小值，返回第一个）。
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    call_oi_arr = np.asarray(call_oi, dtype=float)
    put_oi_arr = np.asarray(put_oi, dtype=float)

    if not (len(strikes_arr) == len(call_oi_arr) == len(put_oi_arr)):
        raise ValueError("strikes / call_oi / put_oi 长度必须一致")
    if len(strikes_arr) == 0:
        raise ValueError("strikes 不能为空")

    payouts = []
    for k in strikes_arr:
        call_payout = np.maximum(k - strikes_arr, 0) * call_oi_arr
        put_payout = np.maximum(strikes_arr - k, 0) * put_oi_arr
        payouts.append(call_payout.sum() + put_payout.sum())

    payouts = np.asarray(payouts)
    return float(strikes_arr[np.argmin(payouts)])


def put_call_ratio(put_volume: float, call_volume: float) -> float:
    """Put/Call Ratio = Put 成交量（或 OI） / Call 成交量（或 OI）。

    该函数对成交量、未平仓合约量（OI）均适用，调用者只需传入对应的数值。
    call_volume 为 0 时返回 inf（意味着当日无 Call 成交，比率无上界）。
    """
    if call_volume == 0:
        return float("inf") if put_volume > 0 else float("nan")
    return put_volume / call_volume


# ---------------------------------------------------------------------------
# 第三批：Black-Scholes 定价与 Greeks
# ---------------------------------------------------------------------------


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T 和 sigma 必须为正数")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes 欧式期权定价（不含分红）。

    Parameters
    ----------
    S : 现价。
    K : 行使价。
    T : 到期时间（年，例如 30 天 = 30/365）。
    r : 无风险利率（年化，小数形式，例如 0.05）。
    sigma : 年化波动率（隐含波动率，小数形式，例如 0.2）。
    option_type : "call" 或 "put"。
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type 必须是 'call' 或 'put'")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Greeks:
    """用 Black-Scholes 公式反推 Greeks：Delta / Gamma / Theta / Vega。

    - Delta：期权价格对现价 S 的一阶偏导，衡量方向敞口。
      Call: N(d1)；Put: N(d1) - 1。
    - Gamma：Delta 对 S 的一阶偏导（Call/Put 相同）：
      φ(d1) / (S * σ * √T)。
    - Theta：期权价格对时间的偏导（这里返回"每年"的 Theta，
      若要"每日"损耗，除以 365 即可）。
    - Vega：期权价格对 σ 的偏导（Call/Put 相同）：S * φ(d1) * √T。

    其中 φ 为标准正态分布的概率密度函数（norm.pdf），N 为标准正态分布的
    累积分布函数（norm.cdf）。

    Returns
    -------
    Greeks(delta, gamma, theta, vega)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
    else:
        raise ValueError("option_type 必须是 'call' 或 'put'")

    return Greeks(delta=float(delta), gamma=float(gamma), theta=float(theta), vega=float(vega))


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """给定期权市场价格，反推隐含波动率（Implied Volatility）。

    用 `scipy.optimize.brentq` 在 `sigma_bounds` 区间内求解使
    `bs_price(..., sigma) == price` 的 sigma。
    """

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - price

    return float(brentq(objective, sigma_bounds[0], sigma_bounds[1], xtol=1e-8))


def iv_skew(
    strikes: list[float],
    ivs: list[float],
    spot: float,
    T: float,
    r: float = 0.0,
) -> dict:
    """IV Skew —— 25-delta 风险逆转（Risk Reversal），规格书 4.6 节口径。

        skew = IV(25-delta 虚值 Put) − IV(25-delta 虚值 Call)

    这里的"25-delta"不是直接选最接近 spot 的行使价，而是要用 Black-Scholes
    反推出 **Call Delta 恰好等于 0.25** 对应的执行价位置（这就是 25-delta
    虚值 Call），以及 **Put Delta 恰好等于 -0.25**（等价于 Call Delta =
    0.75，因为同参数下 put_delta = call_delta - 1）对应的执行价位置（这就
    是 25-delta 虚值 Put）。

    做法：
    1. 用每个行使价自身的 IV，通过 `bs_greeks` 算出该行使价对应的 Call Delta
       （行使价越高，Call Delta 越低，理论上单调递减）。
    2. 按 Delta 升序排列后，用线性插值（`numpy.interp`）分别求出
       Delta=0.25（25-delta Call）与 Delta=0.75（等价于 25-delta Put）处的
       隐含波动率。
    3. skew = IV(Delta=0.75，即25d Put) − IV(Delta=0.25，即25d Call)。

    skew > 0 代表市场为下行保护支付的溢价更高（左偏，常见于股票/股指期权）；
    skew < 0 代表右偏。

    Parameters
    ----------
    strikes : 同一到期日的行使价列表（不要求排序）。
    ivs : 对应行使价的隐含波动率列表。
    spot : 当前标的价格 S。
    T : 剩余年化到期时间，例如 30 天 = 30/365。
    r : 无风险利率（年化，小数形式），默认 0。

    Returns
    -------
    dict，包含 call_25d_iv（25-delta Call 的 IV）、put_25d_iv（25-delta Put
    的 IV）、skew（两者之差）。
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    ivs_arr = np.asarray(ivs, dtype=float)
    if len(strikes_arr) < 2:
        raise ValueError("至少需要 2 个行使价才能定位 25-delta 位置")

    order = np.argsort(strikes_arr)
    strikes_sorted = strikes_arr[order]
    ivs_sorted = ivs_arr[order]

    call_deltas = np.array(
        [
            bs_greeks(spot, k, T, r, iv, "call").delta
            for k, iv in zip(strikes_sorted, ivs_sorted)
        ]
    )

    # Call Delta 随行使价升高而单调递减，np.interp 要求 x 升序，故反转。
    deltas_asc = call_deltas[::-1]
    ivs_asc = ivs_sorted[::-1]

    call_25d_iv = float(np.interp(0.25, deltas_asc, ivs_asc))
    # Put Delta = Call Delta - 1，故 Put Delta = -0.25 等价于 Call Delta = 0.75
    put_25d_iv = float(np.interp(0.75, deltas_asc, ivs_asc))

    return {
        "call_25d_iv": call_25d_iv,
        "put_25d_iv": put_25d_iv,
        "skew": put_25d_iv - call_25d_iv,
    }


def iv_term_structure(expiries_T: list[float], atm_ivs: list[float]) -> dict:
    """IV Term Structure —— 不同到期日 ATM 隐含波动率随期限变化的结构。

    用最小二乘线性回归拟合 atm_iv ~ a + b * T：
    - slope（b）> 0：期限越长 IV 越高，即"正向期限结构"（Contango）；
    - slope（b）< 0：近月 IV 更高，"逆向期限结构"（Backwardation），
      常见于临近重大事件（财报、宏观数据）前。

    同时给出最近月与最远月 IV 的简单比值 near_far_ratio = IV_near / IV_far，
    该比值 < 1 对应 contango，> 1 对应 backwardation。

    Parameters
    ----------
    expiries_T : 各到期日对应的年化剩余期限（升序或任意顺序均可）。
    atm_ivs : 对应到期日的 ATM 隐含波动率。

    Returns
    -------
    dict，包含 slope / intercept / near_far_ratio。
    """
    T_arr = np.asarray(expiries_T, dtype=float)
    iv_arr = np.asarray(atm_ivs, dtype=float)
    if len(T_arr) < 2:
        raise ValueError("至少需要两个到期日才能刻画期限结构")

    slope, intercept = np.polyfit(T_arr, iv_arr, 1)

    order = np.argsort(T_arr)
    near_iv = iv_arr[order][0]
    far_iv = iv_arr[order][-1]
    near_far_ratio = float(near_iv / far_iv) if far_iv != 0 else float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "near_far_ratio": near_far_ratio,
        "structure": "backwardation" if near_far_ratio > 1 else "contango",
    }
