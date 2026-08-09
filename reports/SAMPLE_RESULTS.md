# Alpha Vantage Sample Results

These results come from Alpha Vantage news for Sep 30-Dec 16, 2024 and Yahoo Finance
prices through Dec 20, 2024. The files are committed so the Streamlit app can open with
the project-window workflow already populated. They are educational research artifacts,
not financial advice or evidence of real trading performance.

## Data

- Tickers: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- Alpha Vantage news rows fetched locally: 1,619
- Modeling rows: 355
- Train rows: 266
- Test rows: 89
- Prediction rows: 355, with `split` marking train/test rows
- Label classes: `-1`, `0`, `1`

## Model Comparison

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 0.461 | 0.381 |
| Majority class | 0.517 | 0.227 |
| Stratified random | 0.494 | 0.414 |
| Ticker prior | 0.438 | 0.235 |

## Backtest Summary

Backtest assumptions:

- Initial cash per ticker: $100,000
- Fixed transaction cost: $1.00 per trade
- Slippage: 0.1% per trade
- Backtest rows: held-out `test` split only

| Ticker | Strategy Return | Buy/Hold Return | Trade Count |
| --- | ---: | ---: | ---: |
| AAPL | 2.81% | 6.80% | 7 |
| AMZN | 6.38% | 12.06% | 6 |
| GOOGL | 5.81% | 16.28% | 9 |
| META | 8.60% | 8.16% | 5 |
| MSFT | 3.50% | 5.51% | 7 |
| NVDA | -2.86% | -3.59% | 7 |
| TSLA | 9.67% | 36.75% | 9 |

## Reproduce

```bash
python -m trading_sentiment.cli fetch-alpha-vantage-news --tickers AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META --start 2024-09-30 --end 2024-12-16 --output data/raw/alpha_vantage_news.csv
python -m trading_sentiment.cli fetch-prices --tickers AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META --start 2024-09-30 --end 2024-12-20 --output data/raw/alpha_vantage_prices.csv
python -m trading_sentiment.cli build-dataset --news data/raw/alpha_vantage_news.csv --prices data/raw/alpha_vantage_prices.csv --output data/processed/alpha_vantage_modeling_dataset.csv
python -m trading_sentiment.cli train-baseline --dataset data/processed/alpha_vantage_modeling_dataset.csv --metrics-output reports/alpha_vantage_baseline_metrics.json --predictions-output reports/alpha_vantage_baseline_predictions.csv
python -m trading_sentiment.cli backtest-predictions --predictions reports/alpha_vantage_baseline_predictions.csv --summary-output reports/alpha_vantage_backtest_summary.csv --trades-output reports/alpha_vantage_backtest_trades.csv --equity-output reports/alpha_vantage_backtest_equity_curve.csv --transaction-cost 1.00 --slippage-pct 0.001
```
