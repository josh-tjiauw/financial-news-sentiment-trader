from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trading_sentiment.backtest import backtest_predictions


st.set_page_config(page_title="Financial News Sentiment Trader", layout="wide")

PROJECT_ROOT = Path.cwd()
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "reports" / "baseline_predictions.csv"
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "baseline_metrics.json"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Financial News Sentiment Trader")
st.caption("News-driven stock movement modeling and strategy backtesting")

with st.sidebar:
    st.header("Inputs")
    dataset_path = st.text_input("Modeling dataset CSV", str(DEFAULT_DATASET))
    predictions_path = st.text_input("Predictions CSV", str(DEFAULT_PREDICTIONS))
    initial_cash = st.number_input("Initial cash per ticker", min_value=1000.0, value=100_000.0)
    transaction_cost = st.number_input("Transaction cost per trade", min_value=0.0, value=0.0)

st.header("Pipeline")
st.code(
    "fetch news/prices → build dataset → train baseline → backtest predictions",
    language="text",
)

left, right = st.columns(2)

with left:
    st.subheader("Modeling dataset")
    if Path(dataset_path).exists():
        dataset = load_csv(dataset_path)
        st.metric("Rows", len(dataset))
        if "ticker" in dataset.columns:
            st.metric("Tickers", dataset["ticker"].nunique())
        st.dataframe(dataset.head(25), use_container_width=True)
    else:
        st.info("No modeling dataset found yet. Run `build-dataset` first.")

with right:
    st.subheader("Baseline predictions")
    if Path(predictions_path).exists():
        predictions = load_csv(predictions_path)
        st.metric("Prediction rows", len(predictions))
        if "predicted_label" in predictions.columns:
            st.bar_chart(predictions["predicted_label"].value_counts().sort_index())
        st.dataframe(predictions.head(25), use_container_width=True)
    else:
        predictions = None
        st.info("No predictions found yet. Run `train-baseline` first.")

st.header("Backtest")
if predictions is not None:
    try:
        summary, trades, equity_curve = backtest_predictions(
            predictions,
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
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
