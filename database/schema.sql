-- Financial News Sentiment Trader database schema
-- Initial local research schema, seeded with FXAIX.
--
-- Design goals:
-- - Keep raw market/news data separate from model outputs and trade journal data.
-- - Use stable integer primary keys internally and unique natural keys where useful.
-- - Store money/price values as NUMERIC-compatible decimals rather than floating point.
-- - Track timestamps for future migration into PostgreSQL or another production DB.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS securities (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('stock', 'etf', 'mutual_fund', 'index', 'crypto', 'cash', 'other')),
    exchange TEXT,
    currency TEXT NOT NULL DEFAULT 'USD',
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_bars (
    id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    price_date TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '1d' CHECK (timeframe IN ('1d', '1wk', '1mo')),
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC NOT NULL,
    adjusted_close_price NUMERIC,
    volume INTEGER,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (security_id, price_date, timeframe, source),
    CHECK (high_price IS NULL OR low_price IS NULL OR high_price >= low_price),
    CHECK (open_price IS NULL OR open_price >= 0),
    CHECK (high_price IS NULL OR high_price >= 0),
    CHECK (low_price IS NULL OR low_price >= 0),
    CHECK (close_price >= 0),
    CHECK (adjusted_close_price IS NULL OR adjusted_close_price >= 0),
    CHECK (volume IS NULL OR volume >= 0)
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    published_at TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    raw_text TEXT,
    cleaned_text TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (security_id, published_at, source, title)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    run_name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    training_started_at TEXT,
    training_finished_at TEXT,
    train_start_date TEXT,
    train_end_date TEXT,
    test_start_date TEXT,
    test_end_date TEXT,
    parameters_json TEXT,
    metrics_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    model_run_id INTEGER REFERENCES model_runs(id) ON DELETE SET NULL,
    signal_date TEXT NOT NULL,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('buy', 'hold', 'sell', 'avoid', 'watch')),
    confidence NUMERIC CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    expected_return NUMERIC,
    rationale TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (security_id, model_run_id, signal_date, signal_type)
);

CREATE TABLE IF NOT EXISTS trade_plans (
    id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    plan_date TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    thesis TEXT,
    entry_price NUMERIC,
    stop_loss_price NUMERIC,
    target_price NUMERIC,
    planned_quantity NUMERIC,
    planned_capital NUMERIC,
    max_risk_amount NUMERIC,
    status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'open', 'closed', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (entry_price IS NULL OR entry_price >= 0),
    CHECK (stop_loss_price IS NULL OR stop_loss_price >= 0),
    CHECK (target_price IS NULL OR target_price >= 0),
    CHECK (planned_quantity IS NULL OR planned_quantity >= 0),
    CHECK (planned_capital IS NULL OR planned_capital >= 0),
    CHECK (max_risk_amount IS NULL OR max_risk_amount >= 0)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    trade_plan_id INTEGER REFERENCES trade_plans(id) ON DELETE SET NULL,
    security_id INTEGER NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    quantity NUMERIC NOT NULL CHECK (quantity > 0),
    entry_price NUMERIC NOT NULL CHECK (entry_price >= 0),
    exit_price NUMERIC CHECK (exit_price IS NULL OR exit_price >= 0),
    fees NUMERIC NOT NULL DEFAULT 0 CHECK (fees >= 0),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((status = 'closed' AND closed_at IS NOT NULL AND exit_price IS NOT NULL) OR status IN ('open', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY,
    snapshot_date TEXT NOT NULL UNIQUE,
    cash_balance NUMERIC NOT NULL DEFAULT 0,
    invested_value NUMERIC NOT NULL DEFAULT 0,
    total_value NUMERIC NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (cash_balance >= 0),
    CHECK (invested_value >= 0),
    CHECK (total_value >= 0)
);

CREATE INDEX IF NOT EXISTS idx_price_bars_security_date ON price_bars (security_id, price_date);
CREATE INDEX IF NOT EXISTS idx_news_articles_security_published ON news_articles (security_id, published_at);
CREATE INDEX IF NOT EXISTS idx_signals_security_date ON signals (security_id, signal_date);
CREATE INDEX IF NOT EXISTS idx_trade_plans_security_status ON trade_plans (security_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_security_status ON trades (security_id, status);

INSERT INTO securities (symbol, name, asset_type, exchange, currency)
VALUES ('FXAIX', 'Fidelity 500 Index Fund', 'mutual_fund', 'NASDAQ', 'USD')
ON CONFLICT(symbol) DO UPDATE SET
    name = excluded.name,
    asset_type = excluded.asset_type,
    exchange = excluded.exchange,
    currency = excluded.currency,
    is_active = 1,
    updated_at = CURRENT_TIMESTAMP;
