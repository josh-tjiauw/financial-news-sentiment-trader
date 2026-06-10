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
- [ ] Add deeper historical news provider support

## Phase 3: Modeling

- [ ] Add TF-IDF + logistic regression baseline
- [ ] Add model evaluation report
- [ ] Add FinBERT sentiment scoring experiment
- [ ] Compare model performance against naive baselines

## Phase 4: Backtesting

- [ ] Add benchmark comparison against SPY/QQQ/buy-and-hold
- [ ] Add transaction costs and slippage
- [ ] Add strategy metrics: total return, max drawdown, Sharpe-like risk metric, win rate

## Phase 5: Demo

- [ ] Build Streamlit dashboard
- [ ] Add screenshots/GIF to README
- [ ] Deploy demo if practical
