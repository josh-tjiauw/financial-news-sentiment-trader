from __future__ import annotations

from typing import Any

import pandas as pd

SIGNAL_LABELS = {-1: "Sell", 0: "Hold", 1: "Buy"}


def collect_tickers(*frames: pd.DataFrame | None) -> list[str]:
    """Return sorted tickers from any loaded dashboard frames."""
    tickers: set[str] = set()
    for frame in frames:
        if frame is not None and "ticker" in frame.columns:
            tickers.update(frame["ticker"].dropna().astype(str).unique())
    return sorted(tickers)


def filter_by_tickers(frame: pd.DataFrame, selected_tickers: list[str]) -> pd.DataFrame:
    """Filter a DataFrame by ticker when a ticker column is present."""
    if not selected_tickers or "ticker" not in frame.columns:
        return frame.copy()
    return frame[frame["ticker"].astype(str).isin(selected_tickers)].copy()


def build_metric_comparison(metrics: dict[str, Any]) -> pd.DataFrame:
    """Create a compact model-vs-naive comparison table from metrics JSON."""
    rows: list[dict[str, str | float]] = []

    if "accuracy" in metrics and "macro_f1" in metrics:
        rows.append(
            {
                "model": str(metrics.get("model", "model")),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )

    for name, baseline in metrics.get("naive_baselines", {}).items():
        if "accuracy" not in baseline or "macro_f1" not in baseline:
            continue
        rows.append(
            {
                "model": str(name),
                "accuracy": float(baseline["accuracy"]),
                "macro_f1": float(baseline["macro_f1"]),
            }
        )

    return pd.DataFrame(rows)


def build_metric_score_chart(metric_comparison: pd.DataFrame) -> pd.DataFrame:
    """Shape model metrics for a grouped score chart."""
    if metric_comparison.empty:
        return pd.DataFrame(columns=["model", "metric", "score"])
    return metric_comparison.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )


def build_prediction_signal_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count buy/hold/sell predictions per ticker."""
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


def build_weekly_signal_timeline(
    predictions: pd.DataFrame,
    include_empty_weeks: bool = False,
) -> pd.DataFrame:
    """Aggregate prediction labels into one weekly signal per ticker."""
    required_columns = {"ticker", "date", "predicted_label"}
    if predictions.empty or not required_columns.issubset(predictions.columns):
        return pd.DataFrame(
            columns=["ticker", "week_start", "weekly_signal", "signal_score", "prediction_count"]
        )

    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows["week_start"] = rows["date"] - pd.to_timedelta(rows["date"].dt.weekday, unit="D")
    rows["predicted_label"] = rows["predicted_label"].astype(int)

    weekly = (
        rows.groupby(["ticker", "week_start"], as_index=False)
        .agg(
            signal_score=("predicted_label", "mean"),
            prediction_count=("predicted_label", "size"),
        )
        .sort_values(["week_start", "ticker"])
    )
    weekly["weekly_signal"] = weekly["signal_score"].apply(
        lambda score: "Buy" if score > 0 else "Sell" if score < 0 else "Hold"
    )

    if include_empty_weeks:
        tickers = sorted(rows["ticker"].astype(str).unique())
        week_index = pd.date_range(
            rows["week_start"].min(),
            rows["week_start"].max(),
            freq="W-MON",
        )
        full_index = pd.MultiIndex.from_product(
            [tickers, week_index],
            names=["ticker", "week_start"],
        )
        weekly = (
            weekly.set_index(["ticker", "week_start"])
            .reindex(full_index)
            .reset_index()
            .assign(
                weekly_signal=lambda frame: frame["weekly_signal"].fillna("No signal"),
                prediction_count=lambda frame: frame["prediction_count"].fillna(0).astype(int),
            )
        )

    weekly["week_start"] = weekly["week_start"].dt.date.astype(str)
    return weekly[["ticker", "week_start", "weekly_signal", "signal_score", "prediction_count"]]


def build_return_chart(summary: pd.DataFrame) -> pd.DataFrame:
    """Shape strategy and buy/hold returns for a ticker comparison chart."""
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
    """Average per-ticker equity curves into strategy-vs-buy/hold lines."""
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
