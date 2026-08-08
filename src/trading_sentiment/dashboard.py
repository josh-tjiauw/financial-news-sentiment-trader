from __future__ import annotations

from typing import Any

import pandas as pd


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
