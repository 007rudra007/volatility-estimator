-- SQL schema for Volatility Estimator Supabase Integration
-- Execute this SQL in the Supabase SQL Editor to set up the database.

-- 1. Macro Events Calendar
CREATE TABLE IF NOT EXISTS public.macro_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date DATE NOT NULL,
    event_type TEXT NOT NULL,
    outcome TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_event UNIQUE (date, event_type)
);

-- 2. Volatility Analysis Results
CREATE TABLE IF NOT EXISTS public.volatility_analysis (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    close_price NUMERIC NOT NULL,
    vol_20d NUMERIC,
    vol_52w_avg NUMERIC,
    percentile NUMERIC,
    regime TEXT,
    ewma_vol NUMERIC,
    sma_vol NUMERIC,
    donchian_upper NUMERIC,
    donchian_lower NUMERIC,
    volume_sma NUMERIC,
    rvol NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for volatility analysis search
CREATE INDEX IF NOT EXISTS idx_vol_analysis_ticker_timestamp 
ON public.volatility_analysis (ticker, timestamp DESC);

-- 3. Options GEX Key Levels
CREATE TABLE IF NOT EXISTS public.gex_key_levels (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    spot_price NUMERIC NOT NULL,
    total_net_gex NUMERIC,
    total_net_vex NUMERIC,
    peak_call_strike NUMERIC,
    peak_put_strike NUMERIC,
    peak_net_strike NUMERIC,
    gamma_flip_price NUMERIC,
    gex_regime TEXT,
    vex_regime TEXT,
    gex_at_spot NUMERIC,
    vex_at_spot NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gex_key_levels_ticker_timestamp 
ON public.gex_key_levels (ticker, timestamp DESC);

-- 4. Full Options GEX Profiles (strike-by-strike)
CREATE TABLE IF NOT EXISTS public.gex_profiles (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    strike NUMERIC NOT NULL,
    call_oi NUMERIC,
    put_oi NUMERIC,
    oi NUMERIC,
    gamma NUMERIC,
    vanna NUMERIC,
    call_gex NUMERIC,
    put_gex NUMERIC,
    net_gex NUMERIC,
    call_vex NUMERIC,
    put_vex NUMERIC,
    net_vex NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gex_profiles_ticker_timestamp 
ON public.gex_profiles (ticker, timestamp DESC);

-- 5. Market Positioning (COT / FII net positions)
CREATE TABLE IF NOT EXISTS public.positioning_data (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    speculator_net NUMERIC,
    commercial_net NUMERIC,
    cot_index NUMERIC,
    speculator_52w_min NUMERIC,
    speculator_52w_max NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positioning_ticker_timestamp 
ON public.positioning_data (ticker, timestamp DESC);

-- 6. Cumulative Volume Delta (CVD) Data & Divergence Signals
CREATE TABLE IF NOT EXISTS public.cvd_data (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    close_price NUMERIC,
    volume NUMERIC,
    delta NUMERIC,
    cvd NUMERIC,
    divergence_signal INTEGER, -- +1 = Bullish, -1 = Bearish, 0 = None
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cvd_ticker_timestamp 
ON public.cvd_data (ticker, timestamp DESC);

-- 7. Intraday Next-Day Price Predictions
CREATE TABLE IF NOT EXISTS public.intraday_predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    target_date DATE NOT NULL,
    candle_time TEXT NOT NULL,
    pred_price NUMERIC NOT NULL,
    low_bound NUMERIC NOT NULL,
    high_bound NUMERIC NOT NULL,
    actual_price NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_intraday_pred UNIQUE (ticker, target_date, candle_time)
);

CREATE INDEX IF NOT EXISTS idx_intraday_pred_ticker_target 
ON public.intraday_predictions (ticker, target_date DESC);

-- 8. Quantitative Backtest Results
CREATE TABLE IF NOT EXISTS public.backtest_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    cumulative_return NUMERIC,
    benchmark_return NUMERIC,
    sharpe_ratio NUMERIC,
    max_drawdown NUMERIC,
    win_rate NUMERIC,
    trade_count INTEGER,
    equity_curve JSONB,
    trade_logs JSONB,
    volatility_breaches JSONB,
    cvd_forward_returns JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_backtest_ticker UNIQUE (ticker)
);
