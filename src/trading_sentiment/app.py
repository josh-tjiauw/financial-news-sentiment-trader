from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_sentiment.backtest import backtest_predictions
from trading_sentiment.dashboard import (
    build_metric_comparison,
    collect_tickers,
    filter_by_tickers,
)


st.set_page_config(page_title="Financial News Sentiment Trader", layout="wide")

PROJECT_ROOT = Path.cwd()
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "reports" / "baseline_predictions.csv"
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "baseline_metrics.json"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


st.title("Financial News Sentiment Trader")
st.caption("News-driven stock movement modeling and strategy backtesting")

with st.sidebar:
    st.header("Inputs")
    dataset_path = st.text_input("Modeling dataset CSV", str(DEFAULT_DATASET))
    predictions_path = st.text_input("Predictions CSV", str(DEFAULT_PREDICTIONS))
    metrics_path = st.text_input("Metrics JSON", str(DEFAULT_METRICS))
    initial_cash = st.number_input("Initial cash per ticker", min_value=1000.0, value=100_000.0)
    transaction_cost = st.number_input("Transaction cost per trade", min_value=0.0, value=0.0)
    slippage_pct = st.number_input(
        "Slippage per trade",
        min_value=0.0,
        value=0.0,
        format="%.4f",
    )

dataset = load_csv(str(dataset_path)) if Path(dataset_path).exists() else None
predictions = load_csv(str(predictions_path)) if Path(predictions_path).exists() else None
metrics = load_metrics(str(metrics_path)) if Path(metrics_path).exists() else None
available_tickers = collect_tickers(dataset, predictions)

with st.sidebar:
    selected_tickers = st.multiselect(
        "Tickers",
        available_tickers,
        default=available_tickers,
    )

st.header("Pipeline")
st.code(
    "fetch news/prices -> build dataset -> train baseline -> backtest predictions",
    language="text",
)

if metrics is not None:
    metric_comparison = build_metric_comparison(metrics)
    if not metric_comparison.empty:
        st.header("Model Comparison")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Rows", int(metrics.get("row_count", 0)))
        metric_cols[1].metric("Train rows", int(metrics.get("train_row_count", 0)))
        metric_cols[2].metric("Test rows", int(metrics.get("test_row_count", 0)))
        st.dataframe(metric_comparison, use_container_width=True)
        st.bar_chart(metric_comparison.set_index("model")[["accuracy", "macro_f1"]])
else:
    st.info("No metrics found yet. Run `train-baseline` first.")

left, right = st.columns(2)

with left:
    st.subheader("Modeling dataset")
    if dataset is not None:
        filtered_dataset = filter_by_tickers(dataset, selected_tickers)
        st.metric("Rows", len(filtered_dataset))
        if "ticker" in filtered_dataset.columns:
            st.metric("Tickers", filtered_dataset["ticker"].nunique())
        st.dataframe(filtered_dataset.head(25), use_container_width=True)
    else:
        st.info("No modeling dataset found yet. Run `build-dataset` first.")

with right:
    st.subheader("Baseline predictions")
    if predictions is not None:
        filtered_predictions = filter_by_tickers(predictions, selected_tickers)
        st.metric("Prediction rows", len(filtered_predictions))
        if "predicted_label" in filtered_predictions.columns:
            st.bar_chart(filtered_predictions["predicted_label"].value_counts().sort_index())
        st.dataframe(filtered_predictions.head(25), use_container_width=True)
    else:
        st.info("No predictions found yet. Run `train-baseline` first.")

st.header("Backtest")
if predictions is not None:
    try:
        filtered_predictions = filter_by_tickers(predictions, selected_tickers)
        summary, trades, equity_curve = backtest_predictions(
            filtered_predictions,
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
            slippage_pct=slippage_pct,
        )
        st.subheader("Strategy summary")
        st.dataframe(summary, use_container_width=True)

        if not summary.empty:
            chart_data = summary.set_index("ticker")[["strategy_return", "buy_hold_return"]]
            st.bar_chart(chart_data)

        st.subheader("Equity curve")
        if not equity_curve.empty:
            st.line_chart(equity_curve, x="date", y="equity", color="ticker")
        st.dataframe(equity_curve, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(trades, use_container_width=True)
    except ValueError as exc:
        st.warning(f"Backtest is not ready: {exc}")
else:
    st.info("Backtest results will appear after predictions are available.")

st.header("How to run")
st.code(
    "py -m streamlit run src/trading_sentiment/app.py",
    language="bash",
)
