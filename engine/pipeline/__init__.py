"""编排层：从原始 OHLCV/期权数据到最终 Bull/Bear/Confidence Score 的完整链路。

四个场景各一个端到端函数，对应 `engine.scoring` 的四个模块：

- `run_intraday_pipeline`  <-> `engine.scoring.intraday.score_intraday`
- `run_swing_pipeline`     <-> `engine.scoring.swing.score_swing`
- `run_options_pipeline`   <-> `engine.scoring.options.score_options`
- `run_leveraged_etf_pipeline` <-> `engine.scoring.leveraged_etf.score_leveraged_etf`

每个函数职责一致：读原始数据 -> 调 `engine.indicators` 算指标 -> 调对应
`engine.scoring.score_*` 打分 -> 调 `engine.events.apply_event_adjustments`
算事件调整 -> 合并成 `PipelineResult`。本包不重新实现任何指标公式或打分
逻辑，具体设计原则见 `engine.pipeline.base` 模块文档。
"""
from .base import PipelineResult, combine_with_event_adjustment, redistribute_weights_for_override
from .intraday import run_intraday_pipeline
from .leveraged_etf import run_leveraged_etf_pipeline
from .options import run_options_pipeline
from .swing import run_swing_pipeline

__all__ = [
    "PipelineResult",
    "combine_with_event_adjustment",
    "redistribute_weights_for_override",
    "run_intraday_pipeline",
    "run_swing_pipeline",
    "run_options_pipeline",
    "run_leveraged_etf_pipeline",
]
