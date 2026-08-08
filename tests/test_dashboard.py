import pandas as pd

from trading_sentiment.dashboard import (
    build_metric_comparison,
    collect_tickers,
    filter_by_tickers,
)


def test_collect_tickers_from_multiple_frames():
    dataset = pd.DataFrame({"ticker": ["MSFT", "AAPL"]})
    predictions = pd.DataFrame({"ticker": ["AAPL", "NVDA"]})

    assert collect_tickers(dataset, predictions) == ["AAPL", "MSFT", "NVDA"]


def test_filter_by_tickers_keeps_selected_symbols():
    frame = pd.DataFrame(
        [
            {"ticker": "AAPL", "value": 1},
            {"ticker": "MSFT", "value": 2},
        ]
    )

    filtered = filter_by_tickers(frame, ["MSFT"])

    assert filtered.to_dict("records") == [{"ticker": "MSFT", "value": 2}]


def test_build_metric_comparison_includes_model_and_naive_baselines():
    metrics = {
        "model": "tfidf_logistic_regression",
        "accuracy": 0.75,
        "macro_f1": 0.6,
        "naive_baselines": {
            "majority_class": {"accuracy": 0.5, "macro_f1": 0.33},
            "ticker_prior": {"accuracy": 0.7, "macro_f1": 0.55},
        },
    }

    comparison = build_metric_comparison(metrics)

    assert comparison["model"].tolist() == [
        "tfidf_logistic_regression",
        "majority_class",
        "ticker_prior",
    ]
    assert comparison["accuracy"].tolist() == [0.75, 0.5, 0.7]
