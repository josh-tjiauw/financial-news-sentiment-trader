# Database Schema

This directory contains the first database layer for the trading app.

## Current starting point

- `schema.sql` defines a local SQLite-compatible research schema.
- `trading_sentiment.database` opens the local app database with foreign keys enabled and applies the schema idempotently.
- The CLI command `trading-sentiment init-db` creates/updates `data/app/trading_sentiment.sqlite` by default.
- The schema is seeded with the first tracked security: `FXAIX` / Fidelity 500 Index Fund.

## Core entities

- `securities` — ticker/master data for stocks, ETFs, mutual funds, indexes, crypto, cash, or other instruments.
- `price_bars` — daily/weekly/monthly OHLCV price history by security.
- `news_articles` — normalized news text linked to a security.
- `model_runs` — ML run metadata, parameters, and metrics.
- `signals` — model/manual buy/hold/sell/watch outputs.
- `trade_plans` — planned trade/risk setup before execution.
- `trades` — opened/closed trade journal records.
- `portfolio_snapshots` — periodic portfolio value snapshots.

## Why SQLite-compatible first?

The project is still a local Python research app. SQLite gives us a low-friction schema that can be validated in tests immediately while keeping the table design portable enough to migrate to PostgreSQL later.

## Initialize the local app database

From the repository root:

```bash
py -m trading_sentiment.cli init-db
```

Use `--database path/to/app.sqlite` when you want a throwaway or alternate local database.

## Next suggested schema steps

1. Add import commands to load existing CSV price/news data into `price_bars` and `news_articles`.
2. Add CRUD routes or Streamlit forms for `trade_plans` and `trades`.
3. Add views for dashboard stats: open risk, realized P/L, win rate, and average R multiple.
