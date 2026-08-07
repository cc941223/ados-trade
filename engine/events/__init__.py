"""第 7 章 —— 事件日历集成逻辑：事件不参与打分，只调节 Confidence Score。"""
from .adjustments import EventAdjustment, apply_event_adjustments, combine_multipliers

__all__ = ["EventAdjustment", "apply_event_adjustments", "combine_multipliers"]
