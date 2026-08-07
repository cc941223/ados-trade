"""5.5 节 —— 杠杆 ETF 对冲场景（Regime 判断）：score_leveraged_etf()。

规格书原文强调该场景"本质是判断当前处于 Bull / Chop / Bear 三态中的哪一态
以决定配置比例，而非单纯看涨看跌打分"。为了和其它三个场景保持统一的输出
接口（Bull/Bear/Confidence Score），本函数仍复用同一套聚合逻辑，额外在
`extra["regime"]` 里给出由 Bull/Bear Score 派生的三态标签，供上层直接消费。

子指标、Regime 判断逻辑、V1 初始权重（规格书表 7）：

| 子指标                       | Regime 判断逻辑                             | V1 权重 |
|-------------------------------|-----------------------------------------------|---------|
| VIX 绝对水平                  | ＜15 低波动，15–25 正常，＞25 高波动           | 30%     |
| VIX 期限结构（9D/VIX/3M）      | Backwardation → Bear regime 倾向              | 25%     |
| 大盘 200 日均线位置            | 站上 → Bull 基调；跌破 → Bear 基调            | 25%     |
| 波动损耗率（近 30 天实际值）    | 损耗过高 → 提示减少杠杆敞口                    | 20%     |

本模块不重新计算 VIX 期限结构比值/波动损耗率，原始值需要调用方先用
`engine.indicators.leveraged` 里的函数算好再传入。
"""
from __future__ import annotations

from typing import Optional

from .base import ScoreResult, SubScore, aggregate_scores, linear_map, resolve_weights

DEFAULT_WEIGHTS = {
    "vix_level": 0.30,
    "vix_term_structure": 0.25,
    "ma200_position": 0.25,
    "volatility_decay": 0.20,
}


def classify_regime(bull_score: float, bear_score: float, threshold: float = 10.0) -> str:
    """把 Bull/Bear Score 差值映射成 Bull / Chop / Bear 三态标签。

    差值超过 `threshold`（默认 10 分）才判定为方向明确的 Bull/Bear，否则
    视为 Chop（多空力量接近，倾向横盘震荡，杠杆敞口应保守）。

    ⚠️ V1 占位阈值，待 M3 回测校准：`threshold=10.0` 不是从规格书或历史数据推
    出来的，规格书第 5.5 节只定义了"要分 Bull/Chop/Bear 三态"，没有给出具体
    分界数值。10.0 只是一个能让流程先跑起来的主观默认值，和规格书里 Max Pain
    （"此假设在学术与实务界均有争议，不作为唯一决策依据"）、杠杆 ETF 波动损耗
    近似公式（"假设性质，需第 6 章验证"）性质相同——都属于需要用 M3 walk-forward
    回测框架、拿历史上 Bull-Bear 差值分布去反推"能最好区分事后真趋势与事后真
    横盘"的阈值后再替换的占位值。目前把它做成参数就是为了到时候直接传参覆盖，
    不需要改代码。
    """
    diff = bull_score - bear_score
    if diff > threshold:
        return "bull"
    if diff < -threshold:
        return "bear"
    return "chop"


def score_leveraged_etf(
    price: Optional[float] = None,
    vix: Optional[float] = None,
    vix_term_structure_ratio: Optional[float] = None,
    ma200: Optional[float] = None,
    daily_decay: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    vix_low: float = 15.0,
    vix_high: float = 25.0,
    term_structure_neutral: float = 1.0,
    term_structure_scale: float = 0.15,
    ma_scale: float = 0.05,
    decay_scale: float = 0.005,
    regime_threshold: float = 10.0,
) -> ScoreResult:
    """杠杆 ETF 对冲场景 Bull/Bear/Confidence Score + Regime 标签。

    Parameters
    ----------
    price, ma200 : 大盘（或对应指数）当前价格与 200 日均线。
    vix : VIX 绝对水平。
    vix_term_structure_ratio : 由 `engine.indicators.leveraged.vix_term_structure_ratio`
        算出的 `ratio`（近月/远月，例如 VIX9D/VIX 或 VIX/VIX3M）。
    daily_decay : 由 `engine.indicators.leveraged.volatility_decay_regression`
        算出的 `daily_decay`（近 30 天实际每日波动损耗率，理论上为负）。
    weights : 覆盖 DEFAULT_WEIGHTS 中任意子指标权重的字典。
    vix_low, vix_high : VIX 绝对水平的分段阈值，默认 15 / 25，对应规格书
        "<15 低波动，15-25 正常，>25 高波动"；低于 vix_low 视为满分看多
        （规避杠杆损耗的最佳环境），高于 vix_high 视为满分看空，区间内线性过渡。
        这两个数字**不是占位值**——规格书表 7 原文直接给出了 <15/15-25/>25
        这三段分界，是有依据的，不需要待 M3 校准（除非未来回测发现这组分段
        本身不准确，那是另一个层面的问题）。
    term_structure_neutral : VIX 期限结构比值的中性基准，默认 1.0（近月/远月
        VIX 相等，定义性常量，非占位值——不存在"校准出更好的中性点"这个问题）。
    term_structure_scale, ma_scale, decay_scale : 各子指标线性映射的饱和阈值。

        ⚠️ V1 占位值，待 M3 回测校准：这三个饱和阈值（0.15/0.05/0.005）
        规格书都没有给出具体数字，是能让流程先跑起来的主观默认值，和下面
        `regime_threshold`、`classify_regime` 里的 `threshold=10.0` 性质相同
        ——都需要等 M3 walk-forward 回测框架积累历史数据后再校准，不是从
        规格书或历史分布推出来的。做成参数就是为了到时候直接传参覆盖，不需要
        改代码。
    regime_threshold : 传给 `classify_regime` 的 Bull/Bear 差值阈值——同样是
        V1 占位值，待 M3 回测校准，详见 `classify_regime` 文档。

    Returns
    -------
    ScoreResult(bull_score, bear_score, confidence_score, sub_scores, extra)，
    其中 extra["regime"] 为 "bull" / "chop" / "bear" 三态标签。
    """
    w = resolve_weights(DEFAULT_WEIGHTS, weights)

    # 1. VIX 绝对水平：<vix_low 满分看多（低波动环境对杠杆 ETF 最友好），
    #    >vix_high 满分看空，区间内线性过渡，vix_low/vix_high 之间的中点
    #    对应分数 0（规格书 "15-25 正常" 的中性区间）。
    vix_score = linear_map(vix, vix_low, vix_high, 100, -100)

    # 2. VIX 期限结构：ratio>1（Backwardation）→ Bear regime 倾向，
    #    ratio<1（Contango）→ 正常/偏 Bull，偏离中性值超过 ±term_structure_scale
    #    （默认 0.15）饱和。
    term_structure_raw = (
        (vix_term_structure_ratio - term_structure_neutral) if vix_term_structure_ratio is not None else None
    )
    term_structure_score = linear_map(term_structure_raw, -term_structure_scale, term_structure_scale, 100, -100)

    # 3. 大盘 200 日均线位置：价格相对 200 日均线的偏离幅度，超过 ±ma_scale
    #    （默认 5%）饱和；站上均线 → Bull 基调，跌破 → Bear 基调。
    ma200_position_raw = (price - ma200) / ma200 if (price is not None and ma200) else None
    ma200_score = linear_map(ma200_position_raw, -ma_scale, ma_scale, -100, 100)

    # 4. 波动损耗率：daily_decay 本身已是带符号的每日损耗率（越负代表损耗越
    #    严重），直接线性映射，达到 ±decay_scale（默认 0.5%/天）饱和；
    #    损耗越严重（越负）→ 越应该"减少杠杆敞口"，对应 Bear 方向分数。
    decay_score = linear_map(daily_decay, -decay_scale, decay_scale, -100, 100)

    sub_scores = {
        "vix_level": SubScore("vix_level", vix, vix_score, w["vix_level"]),
        "vix_term_structure": SubScore(
            "vix_term_structure", vix_term_structure_ratio, term_structure_score, w["vix_term_structure"]
        ),
        "ma200_position": SubScore("ma200_position", ma200_position_raw, ma200_score, w["ma200_position"]),
        "volatility_decay": SubScore("volatility_decay", daily_decay, decay_score, w["volatility_decay"]),
    }

    result = aggregate_scores(sub_scores)
    result.extra["regime"] = classify_regime(result.bull_score, result.bear_score, regime_threshold)
    return result
