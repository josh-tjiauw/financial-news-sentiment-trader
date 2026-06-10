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
- Future-return labeling logic
- Basic trading signal generation
- Minimal long/cash backtesting engine
- Unit tests for the core logic
- Project roadmap and architecture notes

## Quick start

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pytest
```

## Example resume bullet

> Built a financial news sentiment analysis platform using Python, NLP, and machine learning to classify market sentiment from news headlines, predict next-day stock movement, and backtest trading strategies against historical price benchmarks.

## Disclaimer

This is an educational software project, not financial advice or a production trading system.
