"""编排层公共基础设施。

四个 `run_*_pipeline()` 函数共享同一套"组装规则"：

    原始 OHLCV/期权数据
        --(engine.indicators)--> 指标原始值
        --(engine.scoring)--> ScoreResult(bull, bear, confidence, sub_scores)
        --(engine.events)--> EventAdjustment(confidence_multiplier, ...)
        --(本文件 combine_with_event_adjustment)--> PipelineResult

本层的定位是纯粹的"调用组装"：不重新实现任何指标公式，也不重新定义任何
打分逻辑，只负责——
1. 从原始 DataFrame 里读数、做必要的对齐/整理（比如取最后一行、按日期
   对齐两条收益率序列），把结果喂给 `engine.indicators` 里已有的函数；
2. 把 `engine.indicators` 算出来的值，按 `engine.scoring` 各 score_* 函数
   要求的参数名对上号；
3. 把 `engine.events.apply_event_adjustments` 算出的调整量，应用到
   `ScoreResult.confidence_score` 上。

这一层如果要引入任何"计算"，都必须是不含指标设计判断的纯数据整理
（取最新值、对齐索引、简单算术），一旦发现某个环节缺的是真正的指标计算
（比如 RSI），就不在这里现造一个，而是作为已知缺口在对应 pipeline 函数的
文档里明确写出来。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from engine.events.adjustments import EventAdjustment
from engine.scoring.base import ScoreResult


def none_if_nan(value: Optional[float]) -> Optional[float]:
    """把 NaN 转成 None。

    `engine.indicators` 里的滑动窗口类函数（`sma`/`atr`/...）在历史数据不足
    一个完整窗口时会返回 NaN（例如只有 10 根 K 线却要算 20 日均线）。
    `engine.scoring` 那边的语义是"传 None 代表该子指标数据缺失，参与完整度
    计算但不参与打分"，不是 NaN——两者不能直接混用（NaN 参与 `linear_map`
    的四则运算和比较会得到不可预期的结果，不会报错也不会被当成缺失处理）。
    这个转换本身不是"指标计算"，只是把两层之间的缺失值语义对齐，所以放在
    编排层做，而不是要求每个 `engine.indicators` 函数都改成返回 None。
    """
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    return value


@dataclass
class PipelineResult:
    """一次端到端编排的最终输出：原始 Bull/Bear Score + 事件调整后的
    Confidence Score + 独立标签，对应规格书 7.1 节"最终展示逻辑"的三段式定义。
    """

    scenario: str
    bull_score: float
    bear_score: float
    raw_confidence_score: float  # 未经事件调整的 Confidence Score（ScoreResult 原值）
    confidence_score: float  # 乘上 event_adjustment.confidence_multiplier 之后的值
    labels: list[str] = field(default_factory=list)
    independent_warnings: list[str] = field(default_factory=list)
    regime: Optional[str] = None  # 仅 leveraged_etf 场景有值（来自 ScoreResult.extra["regime"]）
    sub_scores: dict = field(default_factory=dict)  # ScoreResult.sub_scores 原样传递
    event_adjustments: list[dict] = field(default_factory=list)  # 事件调整审计留痕

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "raw_confidence_score": self.raw_confidence_score,
            "confidence_score": self.confidence_score,
            "labels": self.labels,
            "independent_warnings": self.independent_warnings,
            "regime": self.regime,
            "sub_scores": {k: v.to_dict() for k, v in self.sub_scores.items()},
            "event_adjustments": self.event_adjustments,
        }


def combine_with_event_adjustment(
    scenario: str, score_result: ScoreResult, event_adjustment: EventAdjustment
) -> PipelineResult:
    """把 ScoreResult 和 EventAdjustment 合并成最终的 PipelineResult。

    对应规格书 7.1 节："最终展示逻辑 = 原始 Bull/Bear Score（不变）+ 按事件
    规则调整后的 Confidence Score + 独立的事件提示标签"：
    - Bull/Bear Score 原样保留，事件不参与打分；
    - Confidence Score = 原始值 × confidence_multiplier；
    - labels/independent_warnings 直接从 EventAdjustment 搬过来，跟 Bull/Bear/
      Confidence 三个数值并列展示，不参与任何计算。
    """
    regime = score_result.extra.get("regime") if score_result.extra else None
    return PipelineResult(
        scenario=scenario,
        bull_score=score_result.bull_score,
        bear_score=score_result.bear_score,
        raw_confidence_score=score_result.confidence_score,
        confidence_score=score_result.confidence_score * event_adjustment.confidence_multiplier,
        labels=event_adjustment.labels,
        independent_warnings=event_adjustment.independent_warnings,
        regime=regime,
        sub_scores=score_result.sub_scores,
        event_adjustments=event_adjustment.adjustments,
    )


def redistribute_weights_for_override(
    default_weights: dict[str, float], override_key: str, override_weight: Optional[float]
) -> Optional[dict[str, float]]:
    """把某一个子指标的权重临时改成 `override_weight`，其余子指标权重按原比例
    等比例压缩，使总权重仍为 1.0。

    这是 OpEx 规则"Max Pain 子指标权重从 20% 临时提升至 35%（其余子指标等
    比例压缩）"的具体实现——`apply_event_adjustments` 的文档里写明了"其余
    子指标权重等比例压缩（调用方处理）"，本函数就是那个"调用方"该做的事：
    `engine.events` 只负责算出"这次 Max Pain 权重该是多少"，不负责改权重表
    （它不知道 `engine.scoring` 的权重表长什么样，也不该知道），真正的权重表
    结构在 `engine.scoring.options.DEFAULT_WEIGHTS`，两者的连接点只能在编排层。

    `override_weight=None` 时（没有触发权重调整规则）原样返回 `None`，调用方
    据此判断是否需要覆盖 `score_options` 的 `weights` 参数。

    Parameters
    ----------
    default_weights : 原始权重表（例如 `engine.scoring.options.DEFAULT_WEIGHTS`）。
    override_key : 要覆盖的子指标名（例如 "max_pain_deviation"）。
    override_weight : 新权重；`None` 表示不覆盖。

    Returns
    -------
    新权重表（`override_key` 之外的所有 key 按比例压缩，使总和仍为 1.0），
    或 `None`（未触发覆盖）。
    """
    if override_weight is None:
        return None

    old_weight = default_weights[override_key]
    others_sum = 1.0 - old_weight
    if others_sum <= 0:
        raise ValueError(f"'{override_key}' 的原权重是 {old_weight}，其余子指标权重合计为 0，无法等比例压缩")

    scale = (1.0 - override_weight) / others_sum
    return {k: (override_weight if k == override_key else v * scale) for k, v in default_weights.items()}
