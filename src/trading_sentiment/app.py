from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

from trading_sentiment.backtest import backtest_predictions
from trading_sentiment.dashboard import (
    build_metric_comparison,
    build_weekly_signal_timeline,
    collect_tickers,
    filter_by_tickers,
)


st.set_page_config(page_title="Financial News Sentiment Trader", layout="wide")

DEFAULT_DATASET = Path("data/processed/demo_modeling_dataset.csv")
DEFAULT_PREDICTIONS = Path("reports/demo_baseline_predictions.csv")
DEFAULT_METRICS = Path("reports/demo_baseline_metrics.json")
SIGNAL_LABELS = {-1: "Sell", 0: "Hold", 1: "Buy"}


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def show_grouped_bar_chart(
    chart_data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    ylabel: str,
    percent_axis: bool = False,
) -> None:
    pivot = chart_data.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, width=0.72)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.25)
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_excess_return_chart(summary: pd.DataFrame) -> None:
    if summary.empty or "excess_return" not in summary.columns:
        return

    rows = summary.sort_values("excess_return")
    colors = ["#c2410c" if value < 0 else "#15803d" for value in rows["excess_return"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(rows["ticker"], rows["excess_return"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Strategy return minus buy/hold return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in rows["excess_return"]], padding=4)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_weekly_signal_timeline_chart(weekly_signals: pd.DataFrame) -> None:
    if weekly_signals.empty:
        return

    rows = weekly_signals.copy()
    rows["week_start"] = pd.to_datetime(rows["week_start"])
    tickers = sorted(rows["ticker"].astype(str).unique())
    ticker_positions = {ticker: index for index, ticker in enumerate(tickers)}
    rows["ticker_position"] = rows["ticker"].map(ticker_positions)

    styles = {
        "Buy": {"color": "#15803d", "marker": "^"},
        "Hold": {"color": "#64748b", "marker": "o"},
        "Sell": {"color": "#c2410c", "marker": "v"},
    }

    fig_height = max(4, min(8, 1.0 + len(tickers) * 0.55))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    for signal, style in styles.items():
        signal_rows = rows[rows["weekly_signal"] == signal]
        if signal_rows.empty:
            continue
        ax.scatter(
            signal_rows["week_start"],
            signal_rows["ticker_position"],
            s=80 + (signal_rows["prediction_count"] * 25),
            c=style["color"],
            marker=style["marker"],
            label=signal,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )

    ax.set_yticks(list(ticker_positions.values()))
    ax.set_yticklabels(tickers)
    ax.set_xlabel("Week of news")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Weekly signal", loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.16))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def build_metric_score_chart(metric_comparison: pd.DataFrame) -> pd.DataFrame:
    if metric_comparison.empty:
        return pd.DataFrame(columns=["model", "metric", "score"])
    return metric_comparison.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )


def build_prediction_signal_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "predicted_label"}
    if predictions.empty or not required_columns.issubset(predictions.columns):
        return pd.DataFrame(columns=["ticker", "signal", "count"])

    rows = predictions.copy()
    rows["signal"] = rows["predicted_label"].astype(int).map(SIGNAL_LABELS).fillna("Other")
    return (
        rows.groupby(["ticker", "signal"])
        .size()
        .reset_index(name="count")
        .sort_values(["ticker", "signal"])
    )


def build_latest_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "date", "predicted_label"}
    if predictions.empty or not required_columns.issubset(predictions.columns):
        return pd.DataFrame(columns=["ticker", "date", "signal"])

    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows["signal"] = rows["predicted_label"].astype(int).map(SIGNAL_LABELS).fillna("Other")
    display_columns = ["ticker", "date", "signal"]
    if "prediction_confidence" in rows.columns:
        display_columns.append("prediction_confidence")

    return (
        rows.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)[display_columns]
        .sort_values("ticker")
        .assign(date=lambda frame: frame["date"].dt.date.astype(str))
        .reset_index(drop=True)
    )


def build_return_chart(summary: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "strategy_return", "buy_hold_return"}
    if summary.empty or not required_columns.issubset(summary.columns):
        return pd.DataFrame(columns=["ticker", "series", "return"])

    sort_column = "excess_return" if "excess_return" in summary.columns else "strategy_return"
    sorted_summary = summary.sort_values(sort_column, ascending=False)
    return sorted_summary.melt(
        id_vars="ticker",
        value_vars=["strategy_return", "buy_hold_return"],
        var_name="series",
        value_name="return",
    )


def build_average_equity_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"date", "equity", "buy_hold_equity"}
    if equity_curve.empty or not required_columns.issubset(equity_curve.columns):
        return pd.DataFrame(columns=["date", "series", "equity"])

    averaged = (
        equity_curve.groupby("date")[["equity", "buy_hold_equity"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    chart_data = averaged.melt(
        id_vars="date",
        value_vars=["equity", "buy_hold_equity"],
        var_name="series",
        value_name="value",
    )
    return chart_data.rename(columns={"value": "equity"})


st.title("Financial News Sentiment Trader")
st.caption("News-driven stock movement modeling and strategy backtesting")
st.info(
    "The included demo is intentionally compact and includes mixed-signal headlines. "
    "Use it to inspect the workflow, not to judge real trading performance."
)

with st.sidebar:
    st.header("Dataset")
    data_source = st.selectbox("Source", ["Included demo", "Custom files"])

    if data_source == "Custom files":
        with st.expander("File paths", expanded=True):
            dataset_path = Path(st.text_input("Modeling dataset CSV", str(DEFAULT_DATASET)))
            predictions_path = Path(st.text_input("Predictions CSV", str(DEFAULT_PREDICTIONS)))
            metrics_path = Path(st.text_input("Metrics JSON", str(DEFAULT_METRICS)))
    else:
        dataset_path = DEFAULT_DATASET
        predictions_path = DEFAULT_PREDICTIONS
        metrics_path = DEFAULT_METRICS

dataset = load_csv(str(dataset_path)) if Path(dataset_path).exists() else None
predictions = load_csv(str(predictions_path)) if Path(predictions_path).exists() else None
metrics = load_metrics(str(metrics_path)) if Path(metrics_path).exists() else None
available_tickers = collect_tickers(dataset, predictions)

with st.sidebar:
    st.header("Filters")
    selected_tickers = st.multiselect(
        "Tickers",
        available_tickers,
        default=available_tickers,
    )
    st.header("Backtest")
    initial_cash = st.number_input("Initial cash per ticker", min_value=1000.0, value=100_000.0)
    transaction_cost = st.number_input("Transaction cost per trade", min_value=0.0, value=0.0)
    slippage_pct = st.number_input(
        "Slippage per trade",
        min_value=0.0,
        value=0.0,
        format="%.4f",
    )

filtered_dataset = filter_by_tickers(dataset, selected_tickers) if dataset is not None else None
filtered_predictions = (
    filter_by_tickers(predictions, selected_tickers) if predictions is not None else None
)

st.header("Prediction Signals")
if filtered_predictions is not None:
    signal_cols = st.columns(3)
    signal_cols[0].metric("Prediction rows", len(filtered_predictions))
    signal_cols[1].metric("Tickers", filtered_predictions["ticker"].nunique())
    signal_cols[2].metric("Latest date", str(filtered_predictions["date"].max()))

    latest_signals = build_latest_signals(filtered_predictions)
    weekly_signals = build_weekly_signal_timeline(filtered_predictions)
    if not weekly_signals.empty:
        st.subheader("Weekly Buy/Sell Signal Timeline")
        show_weekly_signal_timeline_chart(weekly_signals)
        st.dataframe(weekly_signals, width="stretch")

    st.subheader("Latest Signal by Ticker")
    st.dataframe(latest_signals, width="stretch")
else:
    st.info("No predictions found yet. Run `train-baseline` first.")

st.header("Backtest")
if filtered_predictions is not None:
    try:
        backtest_input = filtered_predictions
        if "split" in filtered_predictions.columns:
            backtest_input = filtered_predictions[filtered_predictions["split"] == "test"].copy()
        summary, trades, equity_curve = backtest_predictions(
            backtest_input,
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
            slippage_pct=slippage_pct,
        )
        st.subheader("Excess Return vs Buy/Hold")
        if not summary.empty:
            show_excess_return_chart(summary)
            summary_columns = [
                "ticker",
                "strategy_return",
                "buy_hold_return",
                "excess_return",
                "trade_count",
                "max_drawdown",
                "sharpe_like",
            ]
            visible_columns = [column for column in summary_columns if column in summary.columns]
            st.dataframe(summary[visible_columns], width="stretch")

        st.subheader("Average Equity Curve: Strategy vs Buy/Hold")
        if not equity_curve.empty:
            average_equity = build_average_equity_curve(equity_curve)
            st.line_chart(average_equity, x="date", y="equity", color="series")
        st.dataframe(equity_curve, width="stretch")

        st.subheader("Trades")
        st.dataframe(trades, width="stretch")
    except ValueError as exc:
        st.warning(f"Backtest is not ready: {exc}")
else:
    st.info("Backtest results will appear after predictions are available.")

left, right = st.columns(2)

with left:
    st.header("Modeling Dataset")
    if filtered_dataset is not None:
        st.metric("Rows", len(filtered_dataset))
        if "ticker" in filtered_dataset.columns:
            st.metric("Tickers", filtered_dataset["ticker"].nunique())
        st.dataframe(filtered_dataset.head(25), width="stretch")
    else:
        st.info("No modeling dataset found yet. Run `build-dataset` first.")

with right:
    st.header("Model Accuracy")
    if metrics is not None:
        metric_comparison = build_metric_comparison(metrics)
        if not metric_comparison.empty:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Rows", int(metrics.get("row_count", 0)))
            model_row = metric_comparison.iloc[0]
            naive_rows = metric_comparison.iloc[1:]
            best_naive_accuracy = naive_rows["accuracy"].max() if not naive_rows.empty else 0.0
            best_naive_f1 = naive_rows["macro_f1"].max() if not naive_rows.empty else 0.0
            metric_cols[1].metric(
                "Accuracy",
                f"{model_row['accuracy']:.1%}",
                delta=f"{model_row['accuracy'] - best_naive_accuracy:.1%} vs best naive",
            )
            metric_cols[2].metric(
                "Macro F1",
                f"{model_row['macro_f1']:.1%}",
                delta=f"{model_row['macro_f1'] - best_naive_f1:.1%} vs best naive",
            )
            metric_chart = build_metric_score_chart(metric_comparison)
            show_grouped_bar_chart(
                metric_chart,
                index="model",
                columns="metric",
                values="score",
                ylabel="Score",
                percent_axis=True,
            )
            st.dataframe(metric_comparison, width="stretch")
    else:
        st.info("No metrics found yet. Run `train-baseline` first.")
