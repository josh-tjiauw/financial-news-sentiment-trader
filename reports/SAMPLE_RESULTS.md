# Sample Demo Results

These results come from the small committed demo dataset under `data/raw/demo_news.csv` and
`data/raw/demo_prices.csv`. The demo data is intentionally compact, educational, and slightly
ambiguous; it is meant to show the full workflow and dashboard surface, not to support real trading
decisions.

## Demo Data

- Tickers: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- Modeling rows: 91
- Train rows: 68
- Test rows: 23
- Prediction rows: 91, with `split` marking train/test rows
- Label classes: `-1`, `1`

## Model Comparison

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 0.696 | 0.654 |
| Majority class | 0.667 | 0.400 |
| Stratified random | 0.435 | 0.303 |
| Ticker prior | 0.696 | 0.410 |

## Backtest Summary

Backtest assumptions:

- Initial cash per ticker: $100,000
- Fixed transaction cost: $1.00 per trade
- Slippage: 0.1% per trade

| Ticker | Strategy Return | Buy/Hold Return | Trade Count |
| --- | ---: | ---: | ---: |
| AAPL | 1.61% | 4.67% | 2 |
| AMZN | -0.92% | 1.23% | 3 |
| GOOGL | 2.58% | 2.68% | 1 |
| META | 1.88% | 1.98% | 1 |
| MSFT | 0.64% | 0.47% | 3 |
| NVDA | 5.58% | 4.35% | 2 |
| TSLA | 1.75% | 1.38% | 1 |

## Reproduce

```bash
python -m trading_sentiment.cli build-dataset --news data/raw/demo_news.csv --prices data/raw/demo_prices.csv --output data/processed/demo_modeling_dataset.csv
python -m trading_sentiment.cli train-baseline --dataset data/processed/demo_modeling_dataset.csv --metrics-output reports/demo_baseline_metrics.json --predictions-output reports/demo_baseline_predictions.csv
python -m trading_sentiment.cli backtest-predictions --predictions reports/demo_baseline_predictions.csv --summary-output reports/demo_backtest_summary.csv --trades-output reports/demo_backtest_trades.csv --equity-output reports/demo_backtest_equity_curve.csv --transaction-cost 1.00 --slippage-pct 0.001
```
