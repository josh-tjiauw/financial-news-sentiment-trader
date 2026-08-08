# Project Roadmap

## Phase 1: Foundation

- [x] Create package structure
- [x] Add text cleaning utilities
- [x] Add future-return label generation
- [x] Add simple signal generation
- [x] Add minimal backtesting logic
- [x] Add unit tests

## Phase 2: Data pipeline

- [x] Add news ingestion adapter
- [x] Add yfinance price ingestion adapter
- [x] Add dataset builder that joins news and prices by ticker/date
- [x] Store reproducible sample raw datasets
- [x] Add flexible historical news CSV importer
- [ ] Add deeper historical news provider API support

## Phase 3: Modeling

- [x] Add TF-IDF + logistic regression baseline
- [x] Add model evaluation report
- [x] Compare model performance against naive baselines
- [ ] Add FinBERT sentiment scoring experiment

## Phase 4: Backtesting

- [x] Backtest baseline predictions as simple per-ticker long/cash strategies
- [x] Add basic buy-and-hold comparison per ticker
- [x] Add configurable transaction costs
- [x] Add equity curve output
- [x] Add strategy metrics: max drawdown, Sharpe-like risk metric, daily win rate, exposure
- [x] Add completed-trade win/loss metrics
- [x] Add slippage

## Phase 5: Demo

- [x] Build initial Streamlit dashboard
- [ ] Add screenshots/GIF to README
- [ ] Deploy demo if practical
