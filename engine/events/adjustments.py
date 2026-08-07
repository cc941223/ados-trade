"""第 7 章 —— 事件日历集成逻辑。

7.1 节核心设计原则：**事件不参与打分，而是调节 Confidence Score**。
财报/OpEx/除息/再平衡/宏观事件这些日历信息，不作为子指标混入第 5 章的加权表
一起参与 Bull/Bear Score 的计算，而是：

    最终展示逻辑 = 原始 Bull/Bear Score（不变）
                + 按本模块规则调整后的 Confidence Score
                + 独立的事件提示标签

也就是说，本模块只产出"调整量"（置信度乘数 / Max Pain 权重覆盖值 / 标签），
具体怎么把这些调整量应用到 `engine.scoring` 算出来的 `ScoreResult` 上，由
调用方决定（例如 `adjusted_confidence = result.confidence_score *
event_adjustment.confidence_multiplier`），本模块不直接改动 ScoreResult。

⚠️ 规格书 7.3 节原文明确写道："本章所有具体系数（0.5、0.6、0.7、0.85、35%
权重提升等）与第 5 章的 V1 初始权重同属主观设定，需标注为'待验证'，待第 6
章回测框架积累历史数据后，用同样的方式验证这些事件调整规则本身是否真正
提升了评分准确性"——也就是说，规格书里写明的这些具体数字，本身就是待校准
的 V1 占位值，跟 `engine.scoring.leveraged_etf.classify_regime` 里
`threshold=10.0` 的性质完全一样，所以下面每一个系数都标注了同样的
"V1 占位值，待 M3 回测校准"。

本模块不考虑交易所节假日，"交易日"只是排除周末（`numpy.busday_count`
默认周一到周五），不排除感恩节/圣诞节等交易所休市日——这是一个已知的简化，
临近假期的边界判断可能有 1-2 天的误差，等接入真实交易日历后再修正。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from engine.scoring.leveraged_etf import VALID_TARGET_TYPES

#: 四个评分场景的合法取值，与 `engine.scoring` 下四个模块对应。
VALID_SCENARIOS = {"intraday", "swing", "options", "leveraged_etf"}

#: 财报事件规则无条件适用的场景——个股相关场景，跟标的类型无关。
#: leveraged_etf 场景是否适用取决于 target_type（见 apply_event_adjustments
#: 内部的 `earnings_applicable` 计算），不是固定归类，所以单独列在这里，
#: 不直接放进这个集合。
EARNINGS_ALWAYS_APPLICABLE_SCENARIOS = {"intraday", "swing", "options"}


@dataclass
class EventAdjustment:
    """一次事件调整计算的完整输出。"""

    confidence_multiplier: float  # 所有适用事件规则的置信度乘数，已按"叠加方案"合并
    max_pain_weight: Optional[float]  # OpEx 规则触发时的 Max Pain 覆盖权重；未触发为 None
    labels: list[str] = field(default_factory=list)  # 提示性标签，不影响分数
    independent_warnings: list[str] = field(default_factory=list)  # 独立风控警示，不参与 Bull/Bear/Confidence 计算
    adjustments: list[dict] = field(default_factory=list)  # 审计留痕，对应 event_score_adjustments 表的行

    def to_dict(self) -> dict:
        return {
            "confidence_multiplier": self.confidence_multiplier,
            "max_pain_weight": self.max_pain_weight,
            "labels": self.labels,
            "independent_warnings": self.independent_warnings,
            "adjustments": self.adjustments,
        }


def trading_days_between(start: date, end: date) -> int:
    """start 到 end 之间的交易日数（不含节假日调整）。

    `numpy.busday_count(start, end)` 计算半开区间 [start, end) 内的交易日数：
    end 在 start 之后时返回正数（"还有几个交易日到 end"），end 在 start
    之前时返回负数（"end 已经过去几个交易日了"，取绝对值即"事件发生后第几个
    交易日"）。
    """
    return int(np.busday_count(start, end))


def apply_event_adjustments(
    scenario: str,
    current_date: date,
    target_type: str = "broad_index",
    symbol: Optional[str] = None,
    underlying_symbol: Optional[str] = None,
    earnings_date: Optional[date] = None,
    earnings_confirmed: bool = True,
    opex_date: Optional[date] = None,
    is_triple_witching: bool = False,
    ex_dividend_date: Optional[date] = None,
    holds_deep_itm_covered_call: bool = False,
    rebalance_date: Optional[date] = None,
    macro_event_date: Optional[date] = None,
    earnings_window_days: int = 3,
    earnings_confidence_multiplier: float = 0.5,
    opex_window_days: int = 2,
    opex_max_pain_weight: float = 0.35,
    ex_dividend_window_days: int = 3,
    rebalance_window_days: int = 5,
    rebalance_confidence_multiplier: float = 0.7,
    macro_same_day_multiplier: float = 0.6,
    macro_next_day_multiplier: float = 0.85,
) -> EventAdjustment:
    """按 7.2 节规则，把五类日历事件换算成对 Confidence Score / 权重表的调整。

    Parameters
    ----------
    scenario : 当前评分场景，必须是 "intraday"/"swing"/"options"/
        "leveraged_etf" 之一（与 `engine.scoring` 四个模块对应）。不同事件
        规则只对特定场景生效，见下面各参数说明。
    current_date : 当前日期（评分发生的日期）。
    target_type : 仅在 `scenario="leveraged_etf"` 时有意义，取值
        "broad_index"（如 TQQQ/SQQQ 跟踪 QQQ）或 "single_stock"（如
        TSLL/TSLQ 跟踪 TSLA），跟 `engine.scoring.leveraged_etf.score_leveraged_etf`
        用的是同一个概念、同一组取值（`VALID_TARGET_TYPES`）。按规格书 5.5
        节 + 7.2 节的适用范围说明：
        - 财报规则：`target_type="single_stock"` 时适用（按标的个股自身财报
          日期判断）；`target_type="broad_index"` 时不适用（宽基指数不存在
          单一财报日）。
        - 指数再平衡规则：`target_type="broad_index"` 时适用；
          `target_type="single_stock"` 时不适用（再平衡影响的是指数成分和
          权重，与单一个股无关）。
        其余三个场景（intraday/swing/options）不受 `target_type` 影响，
        传默认值即可。
    symbol : 杠杆 ETF/标的自身的代码（例如 "TSLL"），仅用于填充 `adjustments`
        审计记录（对应 `event_score_adjustments` 表的 symbol 列），不参与
        计算逻辑。
    underlying_symbol : `scenario="leveraged_etf"` 且 `target_type=
        "single_stock"` 时，标的所跟踪的个股代码（例如 TSLL 对应 "TSLA"），
        仅用于丰富审计记录里的说明文字，让"这次财报调整依据的是哪个公司的
        财报"可追溯（`symbol` 是 ETF 自己的代码，两者含义不同，不要混用）。
        不提供也不影响计算结果。
    earnings_date, earnings_confirmed : 财报日期与该日期是否为已确认
        （而非 estimated）状态。`scenario="leveraged_etf"` 且
        `target_type="single_stock"` 时，这里传的应该是标的个股自身的财报
        日期（例如 TSLL 传 TSLA 的财报日期），不是 ETF 自己的（ETF 本身没有
        财报）。

        ⚠️ 缺失时的处理方式——不报错，但会留下诊断记录：
        `earnings_date`/`underlying_symbol` 跟 `opex_date` 之类的"通用可选
        字段"性质不一样——它们只在 `target_type="single_stock"` 时才有意义，
        这时候缺失不是"数据暂缺"（不像 `opex_date=None` 那样，任何场景下
        单纯就是没有这个事件），而是**配置矛盾**：调用方选择了
        `target_type="single_stock"`（声明"这是跟踪单一个股的杠杆
        ETF"），却没告诉系统跟踪哪只股票，或没接上那只股票的财报日历。
        因此处理方式跟其它可选字段不同：只要 `target_type="single_stock"`
        且 `earnings_date` 或 `underlying_symbol` 缺失任意一个，就会在
        `adjustments` 里追加一条 `adjustment_type="data_gap_warning"` 的
        记录（`event_type="earnings"`，note 写明具体缺了什么），不抛异常、
        不影响 `confidence_multiplier`/`independent_warnings`（避免这一处
        的配置缺口卡死整条打分流程），但确保这类配置遗漏能在事后审计时被
        发现，而不是彻底沉默。
    opex_date, is_triple_witching : OpEx 到期日，以及该日期是否为 Triple
        Witching（三重魔力日，每年 3/6/9/12 月的第三个星期五）——是否为三重
        魔力日需调用方自行判断并传入，本模块不做日历计算。
    ex_dividend_date, holds_deep_itm_covered_call : 除息日，以及当前是否
        持有该标的的深度价内 Covered Call 仓位。
    rebalance_date : Nasdaq-100 年度指数再平衡日期。
    macro_event_date : 宏观事件（FOMC/CPI/NFP/PCE）发生日期。
    earnings_window_days, opex_window_days, ex_dividend_window_days,
    rebalance_window_days : 各事件"临近"判定的交易日窗口（分别默认 3/2/3/5，
        对应规格书 7.2 节原文）。
    earnings_confidence_multiplier, rebalance_confidence_multiplier,
    macro_same_day_multiplier, macro_next_day_multiplier, opex_max_pain_weight :
        各事件触发后的具体调整系数（分别默认 0.5/0.7/0.6/0.85/0.35）。

        ⚠️ V1 占位值，待 M3 回测校准：以上全部窗口天数与调整系数，规格书 7.3
        节原文明确说明"同属主观设定，需标注为待验证"，跟
        `engine.scoring.leveraged_etf.classify_regime` 里 `threshold=10.0`
        性质相同——即便规格书 7.2 节写出了具体数字，那些数字本身也是待 M3
        回测框架验证是否真的提升了评分准确性的占位值，不是已验证的结论。
        做成参数就是为了到时候直接传参覆盖，不需要改代码。

    Returns
    -------
    EventAdjustment(confidence_multiplier, max_pain_weight, labels,
    independent_warnings, adjustments)

    多事件同时触发时的叠加方案（规格书未定义，见模块级 `combine_multipliers`
    的说明与本函数内部调用方式）。
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario 必须是 {sorted(VALID_SCENARIOS)} 之一，收到：{scenario!r}")
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"target_type 必须是 {sorted(VALID_TARGET_TYPES)} 之一，收到：{target_type!r}")

    # 财报规则的适用范围：intraday/swing/options 无条件适用；leveraged_etf
    # 场景则要看 target_type——single_stock 才适用（按 5.5/7.2 节）。
    earnings_applicable = scenario in EARNINGS_ALWAYS_APPLICABLE_SCENARIOS or (
        scenario == "leveraged_etf" and target_type == "single_stock"
    )
    # 指数再平衡规则的适用范围：仅 leveraged_etf 场景且 target_type=broad_index。
    rebalance_applicable = scenario == "leveraged_etf" and target_type == "broad_index"

    multipliers: list[float] = []
    max_pain_weight: Optional[float] = None
    labels: list[str] = []
    independent_warnings: list[str] = []
    adjustments: list[dict] = []

    def _record(event_type: str, adjustment_type: str, original_value, adjusted_value, note: str) -> None:
        adjustments.append(
            {
                "symbol": symbol,
                "ts": current_date,
                "scenario": scenario,
                "event_type": event_type,
                "adjustment_type": adjustment_type,
                "original_value": original_value,
                "adjusted_value": adjusted_value,
                "note": note,
            }
        )

    # ------------------------------------------------------------------
    # 0. 配置缺口检查：target_type=single_stock 但缺少标的个股财报信息。
    #
    # underlying_symbol/earnings_date 跟 opex_date 之类的"通用可选字段"性质
    # 不一样——它们只在 target_type=single_stock 时才有意义，这时候缺失不是
    # "数据暂缺"，是配置矛盾（选了 single_stock 却没告诉系统跟踪哪只股票，
    # 或没接上那只股票的财报日历）。不做硬性报错（避免调用方还没接好某个
    # 具体标的的财报数据时，整条数据管道被这一处卡死），但必须在
    # `adjustments` 审计记录里留一条明确的诊断条目，确保这类配置遗漏能被
    # 事后追溯发现，不是彻底沉默。
    # ------------------------------------------------------------------
    if scenario == "leveraged_etf" and target_type == "single_stock":
        missing_fields = []
        if earnings_date is None:
            missing_fields.append("标的个股财报日期(earnings_date)")
        if underlying_symbol is None:
            missing_fields.append("标的个股代码(underlying_symbol)")
        if missing_fields:
            _record(
                "earnings",
                "data_gap_warning",
                None,
                None,
                f"target_type=single_stock 但缺少{'、'.join(missing_fields)}，财报规则未生效",
            )

    # ------------------------------------------------------------------
    # 1. 财报日历：适用范围见 earnings_applicable 的计算（intraday/swing/
    #    options 无条件适用；leveraged_etf 仅 target_type=single_stock 时
    #    适用，此时 earnings_date 应为标的个股自身的财报日期）。
    # ------------------------------------------------------------------
    if earnings_date is not None and earnings_applicable:
        days_until = trading_days_between(current_date, earnings_date)
        stock_note = f"（标的个股 {underlying_symbol}）" if scenario == "leveraged_etf" and underlying_symbol else ""

        # 财报前 earnings_window_days 个交易日内（不含财报当天——规格书原文
        # 是"前N个交易日"，严格早于财报日）：Confidence × 0.5。
        if 0 < days_until <= earnings_window_days:
            multipliers.append(earnings_confidence_multiplier)
            labels.append("财报临近，历史规律可能失效")
            _record(
                "earnings",
                "confidence_multiplier",
                1.0,
                earnings_confidence_multiplier,
                f"距财报{stock_note} {days_until} 个交易日（窗口 {earnings_window_days} 天）",
            )

            # leveraged_etf + single_stock 的额外独立警示（规格书 7.2 节新增
            # 一条）：这条警示回答的问题跟 Confidence Score 不是同一件事——
            # Confidence Score 是"方向判断可信不可信"，而这里是"哪怕方向判断
            # 正确，杠杆倍数放大后的波动也可能超出可承受范围"，属于仓位/风险
            # 层面的问题，跟除权除息日的 covered call 警示同属一类：不参与
            # 打分，只作为独立标签单独展示。
            if scenario == "leveraged_etf" and target_type == "single_stock":
                independent_warnings.append(
                    "标的个股财报临近，杠杆敞口风险被放大，建议主动降低仓位或提前平仓"
                )
                _record(
                    "earnings",
                    "independent_flag",
                    None,
                    None,
                    f"距财报{stock_note} {days_until} 个交易日，单票杠杆 ETF 风险放大警示",
                )

        # "日期未确认"标注：规格书 7.2 节原文明确写"不受 3 日窗口限制"，
        # 所以这里只要财报尚未发生（days_until>0）且未确认就标注，不要求
        # 落在 earnings_window_days 窗口内。
        if days_until > 0 and not earnings_confirmed:
            labels.append("日期未确认，谨慎安排到期日")
            _record("earnings", "independent_flag", None, None, f"财报日期{stock_note}为 estimated 状态")

    # ------------------------------------------------------------------
    # 2. OpEx / Triple Witching：仅对期权场景生效（Max Pain 权重调整）。
    # ------------------------------------------------------------------
    if opex_date is not None and scenario == "options":
        days_until = trading_days_between(current_date, opex_date)

        # 距离 OpEx ≤ opex_window_days 个交易日（含 OpEx 当天，即 days_until==0）。
        if 0 <= days_until <= opex_window_days:
            max_pain_weight = opex_max_pain_weight
            _record(
                "opex",
                "weight_shift",
                None,
                opex_max_pain_weight,
                f"距OpEx {days_until} 个交易日（窗口 {opex_window_days} 天），"
                "Max Pain 权重临时提升，其余子指标权重等比例压缩（调用方处理）",
            )

        if days_until == 0 and is_triple_witching:
            labels.append("三重魔力日，Gamma pinning 效应可能加剧")
            _record("opex", "independent_flag", None, None, "Triple Witching 当天")

    # ------------------------------------------------------------------
    # 3. 除权除息日：独立风控警示，不进入 confidence_multiplier，只对期权
    #    场景生效（Covered Call 是期权策略，提前指派风险只在期权场景下有意义）。
    # ------------------------------------------------------------------
    if ex_dividend_date is not None and scenario == "options" and holds_deep_itm_covered_call:
        days_until = trading_days_between(current_date, ex_dividend_date)
        if 0 < days_until <= ex_dividend_window_days:
            independent_warnings.append("提前指派风险")
            _record(
                "ex_dividend",
                "independent_flag",
                None,
                None,
                f"距除息日 {days_until} 个交易日，持有深度价内 Covered Call",
            )

    # ------------------------------------------------------------------
    # 4. 指数再平衡：仅 leveraged_etf 场景且 target_type=broad_index 时生效
    #    （target_type=single_stock 时不适用——再平衡影响的是指数成分和权重，
    #    与单一个股无关）。
    # ------------------------------------------------------------------
    if rebalance_date is not None and rebalance_applicable:
        days_until = trading_days_between(current_date, rebalance_date)
        if 0 <= days_until <= rebalance_window_days:
            multipliers.append(rebalance_confidence_multiplier)
            _record(
                "rebalance",
                "confidence_multiplier",
                1.0,
                rebalance_confidence_multiplier,
                f"距再平衡窗口 {days_until} 个交易日（窗口 {rebalance_window_days} 天）",
            )

    # ------------------------------------------------------------------
    # 5. 宏观事件：所有场景都生效。
    # ------------------------------------------------------------------
    if macro_event_date is not None:
        days_since = trading_days_between(macro_event_date, current_date)
        if days_since == 0:
            multipliers.append(macro_same_day_multiplier)
            _record("macro", "confidence_multiplier", 1.0, macro_same_day_multiplier, "宏观事件当天")
        elif days_since == 1:
            multipliers.append(macro_next_day_multiplier)
            _record("macro", "confidence_multiplier", 1.0, macro_next_day_multiplier, "宏观事件后1个交易日")
        # days_since >= 2（第2个交易日起）或 days_since < 0（事件尚未发生）：不调整。

    combined_multiplier = combine_multipliers(multipliers)

    return EventAdjustment(
        confidence_multiplier=combined_multiplier,
        max_pain_weight=max_pain_weight,
        labels=labels,
        independent_warnings=independent_warnings,
        adjustments=adjustments,
    )


#: combine_multipliers 的下限保护——V1 占位值，见该函数文档。
DEFAULT_MULTIPLIER_FLOOR = 0.1


def combine_multipliers(multipliers: list[float], floor: float = DEFAULT_MULTIPLIER_FLOOR) -> float:
    """多个同时触发的 Confidence 乘数如何叠加——规格书没有定义，这是我的方案。

    方案：**独立概率叠加**——把每条规则的乘数当成一个独立的"这次 Bull/Bear
    Score 仍然可信"的概率估计，多条规则同时触发时，用概率论里独立事件的
    联合概率公式 `P(A∩B) = P(A)×P(B)` 合并（`math.prod` 全部相乘），
    再夹到 `[floor, 1.0]` 区间；空列表返回 1.0（无事件触发，不调整）。

    为什么是"独立概率"而不只是"直觉上的不信任程度打折"：
    - 每条规则回答的是同一个问题的不同侧面——"发生这件事时，历史上这套
      评分方法论还有多大概率仍然适用/可信"。财报窗口、宏观事件、再平衡窗口
      各自对应不同、彼此独立的失效机制（基本面驱动 vs 数据发布瞬间异动 vs
      非基本面资金流），没有理由认为它们会相互抵消或包含，所以按独立事件
      处理、用乘法合并联合概率，是概率论里对"独立不确定性来源"的标准处理
      方式，不是单纯的类比。
    - 这跟 `engine.scoring.base.aggregate_scores` 里 `conviction = strength ×
      agreement`（两个独立维度相乘）是同一种设计语言，本项目已经在用相乘
      表达独立因素的联合效应，这里延续同样的思路。

    floor 下限保护：纯相乘在理论上没有下界，规则数量以后如果增加（比如
    M3 回测发现还有其它值得纳入的事件类型），多个乘数连续相乘可能把
    Confidence 打到接近 0，一个本来还有参考价值的分数会被压成"完全不可信"，
    这在实践中往往过度——" 这次评分不如平时可信"跟"这次评分毫无意义"
    是两个不同的判断，后者需要更强的证据支撑。加一个下限，相当于承认
    "即使叠加再多独立的不确定性来源，Confidence Score 本身作为一个信号，
    也不应该被压到判断力完全消失"。

    ⚠️ V1 占位值，待 M3 回测校准：`floor=0.1` 不是从规格书或历史数据推出来
    的，跟 `classify_regime` 里 `threshold=10.0` 性质相同——是能让流程先跑
    起来的主观默认值。以当前定义的五类规则，同时最多两个乘数相乘（比如
    财报×宏观事件同一天 = 0.5×0.6 = 0.3，或再平衡×宏观事件同一天 =
    0.7×0.6 = 0.42），实际达不到 0.1 这个下限，所以现在加这个参数还不会
    改变任何已有测试用例的结果，纯粹是给未来规则增多时的防御性保护，
    做成参数就是为了到时候直接传参覆盖，不需要改代码。

    也考虑过的替代方案（未采用，如果你更倾向这些，告诉我可以直接换）：
    - 取最小值（只信最严重的一个理由）：会丢失"多个理由同时出现"这个信息，
      财报+宏观事件同一天，跟只有财报单独发生，结果完全一样，直觉上不太对。
    - 加权平均：没有天然的权重来源，还要再造一套主观权重，不如直接相乘简单。

    这个"独立概率叠加+下限保护"方案本身也是我提出的、规格书未定义的设计
    选择，跟其他 V1 占位系数一样，需要你确认是否合理，以及以后 M3 回测框架
    积累数据后重新评估。
    """
    result = 1.0
    for m in multipliers:
        result *= m
    return max(result, floor)
