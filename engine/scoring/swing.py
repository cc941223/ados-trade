"""5.3 节 —— 波段 / 趋势跟踪场景评分：score_swing()。

子指标、方向判断逻辑、V1 初始权重（规格书表 5）：

| 子指标                     | 方向判断逻辑                          | V1 权重 |
|----------------------------|----------------------------------------|---------|
| 相对强度 RS                | RS 线上升 → Bull；下降 → Bear          | 30%     |
| 中长期均线趋势（50/200 日）| 价格站上 → Bull；跌破 → Bear           | 25%     |
| Volume Profile 位置        | 价格在 Value Area 上方 → 偏强；下方 → 偏弱 | 20% |
| Put/Call Ratio             | 极端值时反向解读                       | 15%     |
| 板块 / 大盘联动            | 同向 → 加强；背离 → 降低置信度          | 10%     |

本模块不重新计算 RS/均线/POC/PCR，原始值需要调用方先用
`engine.indicators` 里的函数算好再传入。
"""
from __future__ import annotations

from typing import Optional

from .base import ScoreResult, SubScore, aggregate_scores, linear_map, resolve_weights

DEFAULT_WEIGHTS = {
    "relative_strength": 0.30,
    "ma_trend": 0.25,
    "volume_profile_position": 0.20,
    "put_call_ratio": 0.15,
    "sector_alignment": 0.10,
}


def score_swing(
    price: float,
    rs_value: Optional[float] = None,
    ma50: Optional[float] = None,
    ma200: Optional[float] = None,
    poc: Optional[float] = None,
    put_call_ratio: Optional[float] = None,
    benchmark_return: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    rs_scale: float = 15.0,
    ma_scale: float = 0.05,
    poc_scale: float = 0.03,
    pcr_neutral: float = 1.0,
    pcr_scale: float = 0.5,
    benchmark_scale: float = 0.02,
) -> ScoreResult:
    """波段 / 趋势跟踪场景 Bull/Bear/Confidence Score。

    Parameters
    ----------
    price : 当前价格。
    rs_value : 由 `engine.indicators.trend.relative_strength` 算出的
        Mansfield RS 值（已经是百分比形式，例如 +5 代表比值高于其 N 日均线 5%）。
    ma50, ma200 : 50 日 / 200 日均线。两者若都提供，取相对偏离幅度的平均值；
        只提供一个也可以单独计算。
    poc : 由 `engine.indicators.base.poc` 算出的成交量分布控制点，作为
        Volume Profile 位置的参照（规格书原文用 Value Area 上下沿，这里用
        POC 偏离度近似 —— Value Area 尚未在 indicators 层实现，待补充后
        可替换为更精确的 Value Area 位置判断，不影响本函数整体接口）。
    put_call_ratio : 由 `engine.indicators.options.put_call_ratio` 算出的比率。
    benchmark_return : 同期大盘或板块 ETF 的涨跌幅（小数），代表"板块/大盘联动"。
    weights : 覆盖 DEFAULT_WEIGHTS 中任意子指标权重的字典。
    rs_scale, ma_scale, poc_scale, pcr_scale, benchmark_scale :
        各子指标线性映射的饱和阈值。
    pcr_neutral : Put/Call Ratio 的中性基准值，默认 1.0（Put 与 Call 量相当）。

    Returns
    -------
    ScoreResult(bull_score, bear_score, confidence_score, sub_scores, extra)
    """
    w = resolve_weights(DEFAULT_WEIGHTS, weights)

    # 1. 相对强度 RS：Mansfield RS 本身已是以 0 为中性的百分比偏离值，
    #    达到 ±rs_scale（默认 15）视为饱和满分。
    rs_score = linear_map(rs_value, -rs_scale, rs_scale, -100, 100)

    # 2. 中长期均线趋势：50 日、200 日均线各自算价格偏离幅度后取平均
    #    （只提供一个时就用那一个），偏离超过 ±ma_scale（默认 5%）饱和。
    ma_trend_raw = None
    pct_list = []
    if ma50:
        pct_list.append((price - ma50) / ma50)
    if ma200:
        pct_list.append((price - ma200) / ma200)
    if pct_list:
        ma_trend_raw = sum(pct_list) / len(pct_list)
    ma_trend_score = linear_map(ma_trend_raw, -ma_scale, ma_scale, -100, 100)

    # 3. Volume Profile 位置：价格相对 POC 的偏离幅度，偏离超过 ±poc_scale
    #    （默认 3%）饱和；价格高于 POC → 偏强（正分），低于 POC → 偏弱（负分）。
    volume_profile_raw = (price - poc) / poc if poc else None
    volume_profile_score = linear_map(volume_profile_raw, -poc_scale, poc_scale, -100, 100)

    # 4. Put/Call Ratio：极端值反向解读——PCR 远高于中性值（市场极度看空）
    #    往往是逆向看多信号，远低于中性值则反向看空，偏离超过 ±pcr_scale
    #    （默认 0.5）饱和。
    pcr_raw = (put_call_ratio - pcr_neutral) if put_call_ratio is not None else None
    pcr_score = linear_map(pcr_raw, -pcr_scale, pcr_scale, -100, 100)

    # 5. 板块 / 大盘联动：用大盘/板块同期涨跌幅本身的方向与幅度作为该子指标
    #    的分数（相当于"顺风/逆风"力度），涨跌幅超过 ±benchmark_scale
    #    （默认 2%）饱和；它与资产自身走势方向是否一致，由聚合阶段的
    #    一致性(consistency)计算自动体现——同向时会提升 Confidence Score，
    #    背离时会拉低，无需在此单独处理，对应规格书"同向→加强；背离→降低
    #    置信度"的描述。
    sector_alignment_score = linear_map(benchmark_return, -benchmark_scale, benchmark_scale, -100, 100)

    sub_scores = {
        "relative_strength": SubScore("relative_strength", rs_value, rs_score, w["relative_strength"]),
        "ma_trend": SubScore("ma_trend", ma_trend_raw, ma_trend_score, w["ma_trend"]),
        "volume_profile_position": SubScore(
            "volume_profile_position", volume_profile_raw, volume_profile_score, w["volume_profile_position"]
        ),
        "put_call_ratio": SubScore("put_call_ratio", put_call_ratio, pcr_score, w["put_call_ratio"]),
        "sector_alignment": SubScore(
            "sector_alignment", benchmark_return, sector_alignment_score, w["sector_alignment"]
        ),
    }

    return aggregate_scores(sub_scores)
