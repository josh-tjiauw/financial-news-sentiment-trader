# Sample Demo Results

These results come from the small committed demo dataset under `data/raw/demo_news.csv` and
`data/raw/demo_prices.csv`. The demo data is intentionally compact and educational; it is meant to
show the full workflow and dashboard surface, not to support real trading decisions.

## Demo Data

- Tickers: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- Modeling rows: 35
- Train rows: 26
- Test rows: 9
- Label classes: `-1`, `1`

## Model Comparison

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 1.000 | 1.000 |
| Majority class | 0.667 | 0.400 |
| Stratified random | 0.556 | 0.500 |
| Ticker prior | 0.556 | 0.500 |

## Backtest Summary

Backtest assumptions:

- Initial cash per ticker: $100,000
- Fixed transaction cost: $1.00 per trade
- Slippage: 0.1% per trade

| Ticker | Strategy Return | Buy/Hold Return | Trade Count |
| --- | ---: | ---: | ---: |
| AAPL | 0.00% | 0.00% | 0 |
| AMZN | 0.00% | 0.00% | 0 |
| GOOGL | -0.10% | 0.00% | 1 |
| META | -0.10% | 0.00% | 1 |
| MSFT | -0.10% | 0.00% | 1 |
| NVDA | 6.14% | 6.25% | 1 |
| TSLA | 1.72% | 1.92% | 2 |

## Reproduce

```bash
python -m trading_sentiment.cli build-dataset --news data/raw/demo_news.csv --prices data/raw/demo_prices.csv --output data/processed/demo_modeling_dataset.csv
python -m trading_sentiment.cli train-baseline --dataset data/processed/demo_modeling_dataset.csv --metrics-output reports/demo_baseline_metrics.json --predictions-output reports/demo_baseline_predictions.csv
python -m trading_sentiment.cli backtest-predictions --predictions reports/demo_baseline_predictions.csv --summary-output reports/demo_backtest_summary.csv --trades-output reports/demo_backtest_trades.csv --equity-output reports/demo_backtest_equity_curve.csv --transaction-cost 1.00 --slippage-pct 0.001
```
