"""评分模块公共基础设施。

对应规格书第 5.1 节"设计原则"：

- Bull Score 与 Bear Score 分开计算（而非单一正负分数），保留"多空双弱=震荡"
  与"多空双强"等场景的区分度。
- 每个子指标先标准化为 -100 ~ +100 的可比子分数，再按权重加权求和。
- Confidence Score 不是第三个方向性分数，而是本次 Bull/Bear Score 计算结果的
  可信程度，由数据完整度与子指标间方向一致性共同决定
  （规格书原文还提到第三个因素"当前是否处于历史极端行情"，该因素依赖历史分布
  数据，M3 回测框架积累数据后再补充，V1 版本先只用完整度+一致性两项）。
- V1 阶段权重为主观设定，全部支持从外部传入覆盖，为以后从
  `indicator_correlation_history` 表读数据驱动权重做准备。

标准化方法：V1 统一使用**线性映射法**（`linear_map`），把每个子指标的原始值
按一个可配置的"零点/饱和点"区间线性映射到 -100~100，超出区间则截断
（clip）。这是最简单、最容易解释、且不依赖历史样本的方法，适合数据管道尚未
积累历史分布之前使用；等 M3 回测框架积累足够历史数据后，可以替换成基于历史
分布的百分位映射（percentile mapping）而不需要改动本文件里的聚合逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def linear_map(
    value: Optional[float],
    in_min: float,
    in_max: float,
    out_min: float = -100.0,
    out_max: float = 100.0,
    clip: bool = True,
) -> Optional[float]:
    """线性映射：把 value 从 [in_min, in_max] 区间映射到 [out_min, out_max]。

    `in_min`/`in_max` 也可以反向传入（`in_min` 对应更大的数值），此时映射方向
    自动跟随。超出 [in_min, in_max] 范围的输入，`clip=True` 时会截断到
    out_min/out_max（即视为"已饱和"），这也是本模块统一采用的越界处理方式。

    value 为 None 时直接返回 None（代表该子指标数据缺失，上层按缺失处理）。
    """
    if value is None:
        return None
    if in_max == in_min:
        raise ValueError("in_min 和 in_max 不能相等")

    t = (value - in_min) / (in_max - in_min)
    result = out_min + t * (out_max - out_min)

    if clip:
        lo, hi = (out_min, out_max) if out_min <= out_max else (out_max, out_min)
        result = max(lo, min(hi, result))
    return result


@dataclass
class SubScore:
    """单个子指标的原始值 + 标准化分数 + 权重。

    与规格书 5.1 节"数据留痕要求"对应：`backtest_indicator_contributions`
    表需要同时存原始值与标准化子分数，`to_dict()` 直接产出可写入该表的结构。
    """

    name: str
    raw_value: Optional[float]
    score: Optional[float]  # 标准化后的 -100~100 分数；None 表示该子指标缺失/不适用
    weight: float  # 该子指标的名义权重（0~1），来自权重表，不受缺失影响

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "score": self.score,
            "weight": self.weight,
        }


@dataclass
class ScoreResult:
    """一次评分调用的完整输出。"""

    bull_score: float
    bear_score: float
    confidence_score: float
    sub_scores: dict[str, SubScore]
    extra: dict = field(default_factory=dict)  # 场景相关的附加信息，例如杠杆 ETF 的 regime 标签

    def to_dict(self) -> dict:
        return {
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "confidence_score": self.confidence_score,
            "sub_scores": {k: v.to_dict() for k, v in self.sub_scores.items()},
            "extra": self.extra,
        }


def resolve_weights(default_weights: dict[str, float], overrides: Optional[dict[str, float]]) -> dict[str, float]:
    """合并默认权重表与外部传入的覆盖权重表。

    这是"权重表可以从外部传入覆盖"要求的统一入口：默认用规格书 V1 权重，
    未来把 `indicator_correlation_history` 表里算出的建议权重传进 `overrides`
    即可切换到数据驱动权重，不需要改动 score_* 函数内部逻辑。

    只允许覆盖已知的子指标名（`default_weights` 的 key），未知 key 直接报错，
    避免权重表拼写错误被静默忽略。
    """
    if overrides is None:
        return dict(default_weights)

    unknown = set(overrides) - set(default_weights)
    if unknown:
        raise ValueError(f"未知的子指标权重覆盖项：{sorted(unknown)}")

    merged = dict(default_weights)
    merged.update(overrides)
    return merged


def aggregate_scores(sub_scores: dict[str, SubScore]) -> ScoreResult:
    """把一组 SubScore 聚合成 Bull/Bear/Confidence Score。

    聚合逻辑：
    1. 只用 score 不为 None（即数据可用）的子指标参与计算，权重按可用子指标
       重新归一化（例如某场景 5 个子指标只有 4 个有数据，则用这 4 个的权重
       占比重新算 100%），这样缺失单个子指标不会让总分整体失真。
    2. Bull Score = Σ weight_i × max(score_i, 0) / Σ weight_i（只取每个子指标
       score 里"看多"的部分，即正值部分）。
    3. Bear Score = Σ weight_i × max(-score_i, 0) / Σ weight_i（只取"看空"的
       部分，即负值取绝对值）。
       这样若所有子指标同向看多，Bear Score 会接近 0；若子指标互相矛盾
       （一部分看多一部分看空），Bull 和 Bear 会同时呈现中等值，能区分出
       "多空双强"与"单边行情"，符合 5.1 节设计原则。
    4. 完整度 completeness = 可用子指标权重合计 / 全部子指标权重合计。
    5. 一致性 consistency = |Bull-Bear| / (Bull+Bear)，取值 0~1：
       全部子指标同向时 Bull 和 Bear 一个接近 0，consistency 接近 1；
       多空势均力敌时 Bull≈Bear，consistency 接近 0（矛盾程度最高）。
       没有任何有效信号（Bull=Bear=0）时约定 consistency=1（无信号不算矛盾）。
    6. Confidence Score = (completeness × 0.5 + consistency × 0.5) × 100，
       完整度和一致性各占一半权重，两者任一偏低都会拉低整体置信度。
    """
    total_weight_all = sum(s.weight for s in sub_scores.values())
    available = {k: s for k, s in sub_scores.items() if s.score is not None}
    total_weight_available = sum(s.weight for s in available.values())

    if total_weight_available == 0 or total_weight_all == 0:
        return ScoreResult(bull_score=0.0, bear_score=0.0, confidence_score=0.0, sub_scores=sub_scores)

    bull_score = sum(s.weight * max(s.score, 0.0) for s in available.values()) / total_weight_available
    bear_score = sum(s.weight * max(-s.score, 0.0) for s in available.values()) / total_weight_available

    completeness = total_weight_available / total_weight_all
    total_signal = bull_score + bear_score
    consistency = abs(bull_score - bear_score) / total_signal if total_signal > 0 else 1.0

    confidence_score = (completeness * 0.5 + consistency * 0.5) * 100

    return ScoreResult(
        bull_score=bull_score,
        bear_score=bear_score,
        confidence_score=confidence_score,
        sub_scores=sub_scores,
        extra={"completeness": completeness, "consistency": consistency},
    )
