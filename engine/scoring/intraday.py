"""5.2 节 —— 日内短线场景评分：score_intraday()。

子指标、方向判断逻辑、V1 初始权重（规格书表 4）：

| 子指标                 | 方向判断逻辑                              | V1 权重 |
|------------------------|--------------------------------------------|---------|
| 价格 vs VWAP           | 价格＞VWAP → Bull；价格＜VWAP → Bear        | 25%     |
| RSI                    | RSI＞70 → Bear；RSI＜30 → Bull              | 20%     |
| 量比（当前量/历史均量）| 放量突破方向 → 加强该方向分数               | 25%     |
| 价格相对 ATR 的位置    | 接近日内高点 → 动能强；接近低点 → 动能弱    | 15%     |
| 均线排列（短期 vs 长期）| 多头排列 → Bull；空头排列 → Bear            | 15%     |

本模块不重新计算 VWAP/RSI/ATR/均线，这些原始值需要调用方先用
`engine.indicators` 里的函数算好再传入。
"""
from __future__ import annotations

from typing import Optional

from .base import ScoreResult, SubScore, aggregate_scores, linear_map, resolve_weights

DEFAULT_WEIGHTS = {
    "price_vs_vwap": 0.25,
    "rsi": 0.20,
    "volume_ratio": 0.25,
    "atr_position": 0.15,
    "ma_alignment": 0.15,
}


def score_intraday(
    price: float,
    vwap: Optional[float] = None,
    rsi: Optional[float] = None,
    volume_ratio: Optional[float] = None,
    price_change_pct: Optional[float] = None,
    atr: Optional[float] = None,
    reference_price: Optional[float] = None,
    short_ma: Optional[float] = None,
    long_ma: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    vwap_scale: float = 0.01,
    volume_ratio_saturation: float = 2.0,
    atr_z_scale: float = 2.0,
    ma_scale: float = 0.01,
) -> ScoreResult:
    """日内短线场景 Bull/Bear/Confidence Score。

    Parameters
    ----------
    price : 当前价格。
    vwap : 由 `engine.indicators.base.vwap` 算出的当前 VWAP。
    rsi : 0~100 的 RSI 值（调用方自行计算，本模块不实现 RSI）。
    volume_ratio : 当前量 / 历史均量（例如 5 日或 20 日均量）。
    price_change_pct : 当日涨跌幅（小数），用来判断"放量"应该加强哪个方向——
        量比本身没有方向，必须配合价格变动方向才能知道是"放量上涨"还是
        "放量下跌"；缺省为 None 时，量比子指标视为数据不足，不计入本次评分。
    atr : 由 `engine.indicators.trend.atr` 算出的 ATR。
    reference_price : 计算"价格相对 ATR 位置"的基准价（例如当日开盘价），
        z = (price - reference_price) / atr。
    short_ma, long_ma : 短周期与长周期均线值（例如 5 日 vs 20 日）。
    weights : 覆盖 DEFAULT_WEIGHTS 中任意子指标权重的字典，未提供的 key 沿用默认值。
    vwap_scale, volume_ratio_saturation, atr_z_scale, ma_scale :
        各子指标线性映射的饱和阈值，可按需调参（详见各子指标计算注释）。

    Returns
    -------
    ScoreResult(bull_score, bear_score, confidence_score, sub_scores, extra)
    """
    w = resolve_weights(DEFAULT_WEIGHTS, weights)

    # 1. 价格 vs VWAP：涨跌幅超过 ±vwap_scale（默认 1%）即视为饱和满分。
    price_vs_vwap_raw = (price - vwap) / vwap if vwap else None
    price_vs_vwap_score = linear_map(price_vs_vwap_raw, -vwap_scale, vwap_scale, -100, 100)

    # 2. RSI：0~100 整体线性反向映射，50 为中性零点，0→+100（超卖看多），
    #    100→-100（超买看空），自然覆盖 "RSI>70 看空 / RSI<30 看多" 的方向。
    rsi_score = linear_map(rsi, 0, 100, 100, -100)

    # 3. 量比：只在放量（ratio>1）且已知价格方向时才产生分数，
    #    ratio 达到 volume_ratio_saturation（默认 2 倍均量）时饱和满分；
    #    缩量（ratio<=1）不构成"放量突破"，记 0 分（既不加强多也不加强空）。
    volume_ratio_score = None
    if volume_ratio is not None and price_change_pct is not None:
        magnitude = linear_map(volume_ratio, 1.0, volume_ratio_saturation, 0.0, 100.0)
        direction = 1.0 if price_change_pct >= 0 else -1.0
        volume_ratio_score = direction * magnitude

    # 4. 价格相对 ATR 的位置：用 ATR 把价格偏离基准价的距离标准化成"几个 ATR"，
    #    z 达到 ±atr_z_scale（默认 2 个 ATR）视为饱和满分。
    atr_position_raw = None
    atr_position_score = None
    if atr is not None and reference_price is not None and atr > 0:
        atr_position_raw = (price - reference_price) / atr
        atr_position_score = linear_map(atr_position_raw, -atr_z_scale, atr_z_scale, -100, 100)

    # 5. 均线排列：短期均线相对长期均线的偏离幅度，超过 ±ma_scale（默认 1%）饱和。
    ma_alignment_raw = None
    ma_alignment_score = None
    if short_ma is not None and long_ma is not None and long_ma != 0:
        ma_alignment_raw = (short_ma - long_ma) / long_ma
        ma_alignment_score = linear_map(ma_alignment_raw, -ma_scale, ma_scale, -100, 100)

    sub_scores = {
        "price_vs_vwap": SubScore("price_vs_vwap", price_vs_vwap_raw, price_vs_vwap_score, w["price_vs_vwap"]),
        "rsi": SubScore("rsi", rsi, rsi_score, w["rsi"]),
        "volume_ratio": SubScore("volume_ratio", volume_ratio, volume_ratio_score, w["volume_ratio"]),
        "atr_position": SubScore("atr_position", atr_position_raw, atr_position_score, w["atr_position"]),
        "ma_alignment": SubScore("ma_alignment", ma_alignment_raw, ma_alignment_score, w["ma_alignment"]),
    }

    return aggregate_scores(sub_scores)
