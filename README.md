# Financial News Sentiment Trader

AI stocks class project: a Python machine learning pipeline that analyzes financial news, predicts next-day stock movement, and backtests trading strategies against market benchmarks.

## Project goal

Use financial news headlines/summaries to estimate market sentiment, convert that signal into trading decisions, and evaluate whether the strategy performs better than simple benchmarks.

## Planned pipeline

1. Collect financial news by ticker/date.
2. Collect historical stock prices.
3. Clean and normalize article text.
4. Generate labels from future price movement, not same-day movement.
5. Train baseline NLP models.
6. Add finance-specific sentiment modeling, such as FinBERT.
7. Backtest strategy with realistic assumptions.
8. Visualize model results, trades, and benchmark comparison.

## Current foundation

This repo currently includes:

- Python package structure under `src/trading_sentiment`
- Typed domain models for news articles, price bars, signals, trades, and portfolio state
- Text-cleaning utilities
- News CSV ingestion
- Yahoo Finance price ingestion through `yfinance`
- Dataset builder that joins news and prices by ticker/date
- Future-return labeling logic
- Basic trading signal generation
- Minimal long/cash backtesting engine
- CLI command: `trading-sentiment`
- Unit tests for the core logic
- Project roadmap and architecture notes

## Quick start

From the repository root:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
py -m pytest
py -m ruff check .
```

If you do not want to create a virtual environment yet, you can run it with your existing Python install:

```bash
py -m pip install -e .[dev]
py -m pytest
```

## Run the current data pipeline

### 1. Build a modeling dataset from the included sample files

```bash
trading-sentiment build-dataset --news data/raw/sample_news.csv --prices data/raw/sample_prices.csv --output data/processed/sample_modeling_dataset.csv
```

Equivalent module form:

```bash
py -m trading_sentiment.cli build-dataset --news data/raw/sample_news.csv --prices data/raw/sample_prices.csv --output data/processed/sample_modeling_dataset.csv
```

This creates a processed CSV with:

- aggregated news per ticker/date
- combined article text
- cleaned model-ready text
- stock OHLCV price fields
- future close price
- future return
- movement label: `1`, `0`, or `-1`

### 2. Fetch real historical prices

```bash
trading-sentiment fetch-prices --tickers AAPL,MSFT,NVDA --start 2024-01-01 --end 2024-03-01 --output data/raw/prices.csv
```

### 3. Provide real news data

Create a CSV with these columns:

```csv
ticker,published_date,title,summary,source
AAPL,2024-01-02,Apple shares rise after services growth,Analysts noted stronger demand,sample
```

Then build the dataset:

```bash
trading-sentiment build-dataset --news data/raw/news.csv --prices data/raw/prices.csv --output data/processed/modeling_dataset.csv
```

## Example resume bullet

> Built a financial news sentiment analysis platform using Python, NLP, and machine learning to classify market sentiment from news headlines, predict next-day stock movement, and backtest trading strategies against historical price benchmarks.

## Disclaimer

This is an educational software project, not financial advice or a production trading system.
