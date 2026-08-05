-- ADOS-Trade 数据库表结构
-- 基于 TimescaleDB (PostgreSQL 扩展)
-- 对应技术规格书 第3章(数据源)、第4章(自算指标)

-- 启用 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =========================================
-- 1. 正股 OHLCV（IBKR 数据源）
-- =========================================
CREATE TABLE stock_ohlcv (
    symbol      TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    timeframe   TEXT NOT NULL,          -- '1min' / '5min' / '1day' 等
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    volume      BIGINT,
    PRIMARY KEY (symbol, ts, timeframe)
);
SELECT create_hypertable('stock_ohlcv', 'ts');
CREATE INDEX idx_stock_ohlcv_symbol ON stock_ohlcv (symbol, timeframe, ts DESC);

-- =========================================
-- 2. 期权链结构（IBKR 数据源，变化不频繁）
-- =========================================
CREATE TABLE option_contracts (
    contract_id     BIGINT PRIMARY KEY,
    underlying      TEXT NOT NULL,
    expiration_date DATE NOT NULL,
    strike          NUMERIC NOT NULL,
    right           TEXT NOT NULL CHECK (right IN ('C', 'P')),
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_option_contracts_underlying ON option_contracts (underlying, expiration_date, strike);

-- =========================================
-- 3. 期权行情快照（IBKR 数据源，含 IV / Greeks）
-- Greeks 字段对应 CPAPI field code: 7308 Delta / 7309 Gamma / 7310 Theta / 7311 Vega
-- =========================================
CREATE TABLE option_snapshot (
    contract_id     BIGINT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    last_price      NUMERIC,
    bid             NUMERIC,
    ask             NUMERIC,
    volume          BIGINT,
    open_interest   BIGINT,
    implied_vol     NUMERIC,        -- 年化 IV
    delta           NUMERIC,        -- 若 CPAPI 字段可用直接存；否则由 Engine 用 IV 反推填入
    gamma           NUMERIC,
    theta           NUMERIC,
    vega            NUMERIC,
    PRIMARY KEY (contract_id, ts)
);
SELECT create_hypertable('option_snapshot', 'ts');
CREATE INDEX idx_option_snapshot_contract ON option_snapshot (contract_id, ts DESC);

-- =========================================
-- 4. IV 百分位（IBKR 现成字段，13/26/52周）
-- =========================================
CREATE TABLE iv_percentile (
    symbol      TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    period      TEXT NOT NULL,      -- '13w' / '26w' / '52w'
    percentile  NUMERIC,
    PRIMARY KEY (symbol, ts, period)
);
SELECT create_hypertable('iv_percentile', 'ts');

-- =========================================
-- 5. 财报日历 + 历史意外幅度（Finnhub 免费层）
-- =========================================
CREATE TABLE earnings_calendar (
    symbol              TEXT NOT NULL,
    report_date         DATE NOT NULL,
    time_of_day         TEXT,           -- 'BMO'(盘前) / 'AMC'(盘后) / NULL(未知)
    is_confirmed        BOOLEAN DEFAULT FALSE,   -- 是否为官方确认日期，而非预估
    eps_estimate        NUMERIC,
    eps_actual          NUMERIC,
    eps_surprise_pct    NUMERIC,
    revenue_estimate    NUMERIC,
    revenue_actual      NUMERIC,
    revenue_surprise_pct NUMERIC,
    actual_move_pct     NUMERIC,        -- 财报后次日实际涨跌幅，用于验证期权评分有效性
    PRIMARY KEY (symbol, report_date)
);

-- =========================================
-- 6. 期权到期日历（OpEx / Triple Witching，官方公开规则）
-- =========================================
CREATE TABLE opex_calendar (
    opex_date           DATE PRIMARY KEY,
    is_monthly           BOOLEAN DEFAULT TRUE,
    is_triple_witching   BOOLEAN DEFAULT FALSE
);

-- =========================================
-- 7. 除权除息日历（IBKR 合约信息 / 公司 IR）
-- =========================================
CREATE TABLE dividend_calendar (
    symbol          TEXT NOT NULL,
    ex_date         DATE NOT NULL,
    amount          NUMERIC,
    PRIMARY KEY (symbol, ex_date)
);

-- =========================================
-- 8. 指数再平衡公告（Nasdaq 官网）
-- =========================================
CREATE TABLE index_rebalance_calendar (
    index_name      TEXT NOT NULL,
    rebalance_date  DATE NOT NULL,
    description     TEXT,
    PRIMARY KEY (index_name, rebalance_date)
);

-- =========================================
-- 9. 宏观事件日历（美联储官网 / BLS 官网）
-- =========================================
CREATE TABLE macro_events (
    event_type      TEXT NOT NULL,      -- 'FOMC' / 'CPI' / 'NFP' / 'PCE'
    event_date      TIMESTAMPTZ NOT NULL,
    description     TEXT,
    PRIMARY KEY (event_type, event_date)
);

-- =========================================
-- 10. Short Interest（FINRA，延迟约两周）
-- =========================================
CREATE TABLE short_interest (
    symbol          TEXT NOT NULL,
    report_date     DATE NOT NULL,
    short_shares    BIGINT,
    days_to_cover   NUMERIC,
    PRIMARY KEY (symbol, report_date)
);

-- =========================================
-- 11. 分析师评级 / 目标价（Finnhub 免费层）
-- =========================================
CREATE TABLE analyst_ratings (
    symbol          TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    firm            TEXT,
    rating          TEXT,
    price_target    NUMERIC,
    PRIMARY KEY (symbol, ts, firm)
);

-- =========================================
-- 12. 社交媒体情绪（Finnhub 免费层，Reddit/Twitter MSPR）
-- =========================================
CREATE TABLE social_sentiment (
    symbol          TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL,      -- 'reddit' / 'twitter'
    mention_count   INTEGER,
    positive_score  NUMERIC,
    negative_score  NUMERIC,
    mspr            NUMERIC,
    PRIMARY KEY (symbol, ts, source)
);
SELECT create_hypertable('social_sentiment', 'ts');

-- =========================================
-- 13. ADOS Engine 计算结果（自算指标，第4章）
-- 通用宽表，每个指标一行，便于扩展新指标而不改表结构
-- =========================================
CREATE TABLE engine_indicators (
    symbol          TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    indicator_name  TEXT NOT NULL,      -- 'vwap' / 'max_pain' / 'atr' / 'bull_score' / 'gex' 等
    value           NUMERIC,
    metadata        JSONB,              -- 存放该指标的额外参数/中间计算值
    PRIMARY KEY (symbol, ts, indicator_name)
);
SELECT create_hypertable('engine_indicators', 'ts');
CREATE INDEX idx_engine_indicators_lookup ON engine_indicators (symbol, indicator_name, ts DESC);

-- =========================================
-- 数据保留策略（可选，视磁盘空间决定是否启用）
-- 示例：分钟线数据保留 90 天后自动压缩
-- =========================================
-- SELECT add_compression_policy('stock_ohlcv', INTERVAL '90 days');
