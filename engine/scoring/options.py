"""5.4 节 —— 期权策略场景评分：score_options()。

子指标、方向判断逻辑、V1 初始权重（规格书表 6）：

| 子指标                             | 方向判断逻辑                                              | V1 权重 |
|-------------------------------------|-------------------------------------------------------------|---------|
| IV Skew                             | 正 Skew 越强 → 市场偏空头保护，提示谨慎                     | 25%     |
| IV Term Structure                    | Backwardation → 短期不确定性高，提示降低仓位                | 20%     |
| Max Pain vs 当前价                   | 偏离越大 → 回归压力提示越强                                 | 20%     |
| IV Percentile                        | 高 IV Rank → 适合卖方策略；低 IV Rank → 适合买方策略（影响策略选择而非方向）| 20% |
| 历史财报意外幅度（仅财报窗口期）      | 历史持续超预期 → 提示 Bull 倾向                              | 15%     |

本模块不重新计算 IV Skew / IV Term Structure / Max Pain，原始值需要调用方
先用 `engine.indicators.options` 里的函数算好再传入。
"""
from __future__ import annotations

from typing import Optional

from .base import ScoreResult, SubScore, aggregate_scores, linear_map, resolve_weights

DEFAULT_WEIGHTS = {
    "iv_skew": 0.25,
    "iv_term_structure": 0.20,
    "max_pain_deviation": 0.20,
    "iv_percentile": 0.20,
    "earnings_surprise": 0.15,
}


def score_options(
    price: float,
    iv_skew: Optional[float] = None,
    iv_term_structure_ratio: Optional[float] = None,
    max_pain: Optional[float] = None,
    iv_percentile: Optional[float] = None,
    earnings_surprise_avg: Optional[float] = None,
    in_earnings_window: bool = False,
    weights: Optional[dict[str, float]] = None,
    iv_skew_scale: float = 0.05,
    term_structure_neutral: float = 1.0,
    term_structure_scale: float = 0.15,
    max_pain_scale: float = 0.03,
    earnings_surprise_scale: float = 0.05,
) -> ScoreResult:
    """期权策略场景 Bull/Bear/Confidence Score。

    Parameters
    ----------
    price : 标的当前价格。
    iv_skew : 由 `engine.indicators.options.iv_skew` 算出的
        `put_25d_iv - call_25d_iv`（25-delta 风险逆转值）。
    iv_term_structure_ratio : 由 `engine.indicators.options.iv_term_structure`
        算出的 `near_far_ratio`（近月 IV / 远月 IV）。
    max_pain : 由 `engine.indicators.options.max_pain` 算出的最大痛点行使价。
    iv_percentile : IV 百分位（0~100），仅影响策略选择（卖方 vs 买方），
        规格书明确其"影响策略选择而非方向"，因此本函数强制其方向分数为 0，
        不贡献 Bull/Bear 分数，但仍计入权重表（供 `backtest_indicator_contributions`
        留痕，也计入完整度计算）。
    earnings_surprise_avg : 历史财报 EPS/营收 surprise 的平均幅度（小数），
        仅在 `in_earnings_window=True`（当前处于财报窗口期）时才计入评分，
        窗口期外该子指标视为不适用（等同缺失，权重重新归一化）。
    weights : 覆盖 DEFAULT_WEIGHTS 中任意子指标权重的字典。
    iv_skew_scale, term_structure_scale, max_pain_scale, earnings_surprise_scale :
        各子指标线性映射的饱和阈值。

        ⚠️ V1 占位值，待 M3 回测校准：这四个饱和阈值（0.05/0.15/0.03/0.05）
        规格书都没有给出具体数字，是能让流程先跑起来的主观默认值，和
        `classify_regime` 里 `threshold=10.0` 性质相同——都需要等 M3
        walk-forward 回测框架积累历史数据后再校准，不是从规格书或历史分布
        推出来的。做成参数就是为了到时候直接传参覆盖，不需要改代码。
    term_structure_neutral : IV Term Structure 比值的中性基准，默认 1.0
        （近月 IV 等于远月 IV，定义性常量，非占位值——不存在"校准出更好的
        中性点"这个问题，比值本身就该以 1.0 为中性）。

    Returns
    -------
    ScoreResult(bull_score, bear_score, confidence_score, sub_scores, extra)
    """
    w = resolve_weights(DEFAULT_WEIGHTS, weights)

    # 1. IV Skew：正 Skew（Put IV 高于 Call IV）越强代表市场为下行保护付出的
    #    溢价越高，提示谨慎/偏空，因此符号取反——skew 越正，分数越负。
    iv_skew_score = linear_map(iv_skew, -iv_skew_scale, iv_skew_scale, 100, -100)

    # 2. IV Term Structure：ratio = 近月IV/远月IV，>1 为 Backwardation
    #    （短期不确定性高，提示降低仓位/偏谨慎），<1 为正常 Contango。
    #    同样取反：ratio 越高于中性值，分数越负。
    term_structure_raw = (
        (iv_term_structure_ratio - term_structure_neutral) if iv_term_structure_ratio is not None else None
    )
    iv_term_structure_score = linear_map(term_structure_raw, -term_structure_scale, term_structure_scale, 100, -100)

    # 3. Max Pain vs 当前价：Max Pain 高于现价 → 上方存在"回归压力"提示看多；
    #    Max Pain 低于现价 → 下方回归压力提示看空。偏离超过 ±max_pain_scale
    #    （默认 3%）饱和。
    max_pain_deviation_raw = (max_pain - price) / price if max_pain is not None else None
    max_pain_score = linear_map(max_pain_deviation_raw, -max_pain_scale, max_pain_scale, -100, 100)

    # 4. IV Percentile：规格书明确"影响策略选择而非方向"，方向分数强制为 0；
    #    提供了数值就视为"可用"（计入完整度），只是不产生 Bull/Bear 贡献。
    iv_percentile_score = 0.0 if iv_percentile is not None else None

    # 5. 历史财报意外幅度：仅财报窗口期内生效，窗口期外视为不适用（None）。
    earnings_surprise_score = None
    if in_earnings_window and earnings_surprise_avg is not None:
        earnings_surprise_score = linear_map(
            earnings_surprise_avg, -earnings_surprise_scale, earnings_surprise_scale, -100, 100
        )

    sub_scores = {
        "iv_skew": SubScore("iv_skew", iv_skew, iv_skew_score, w["iv_skew"]),
        "iv_term_structure": SubScore(
            "iv_term_structure", iv_term_structure_ratio, iv_term_structure_score, w["iv_term_structure"]
        ),
        "max_pain_deviation": SubScore(
            "max_pain_deviation", max_pain_deviation_raw, max_pain_score, w["max_pain_deviation"]
        ),
        "iv_percentile": SubScore("iv_percentile", iv_percentile, iv_percentile_score, w["iv_percentile"]),
        "earnings_surprise": SubScore(
            "earnings_surprise", earnings_surprise_avg, earnings_surprise_score, w["earnings_surprise"]
        ),
    }

    return aggregate_scores(sub_scores)
