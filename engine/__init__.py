"""ADOS Engine —— 指标计算层。

按技术规格书第4章分批实现：
1. 通用基础：VWAP / Volume Profile / Max Pain / Put-Call Ratio
2. 波段趋势：ATR / 相对强度 RS
3. 期权：Black-Scholes Greeks / IV Skew / IV Term Structure
4. 杠杆 ETF：波动损耗率回归 / VIX 期限结构比值
"""
