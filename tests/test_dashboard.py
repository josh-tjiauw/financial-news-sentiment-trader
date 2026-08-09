import pandas as pd

from trading_sentiment.dashboard import (
    build_average_equity_curve,
    build_metric_comparison,
    build_metric_score_chart,
    build_prediction_signal_counts,
    build_return_chart,
    build_weekly_signal_timeline,
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


def test_build_metric_score_chart_melts_accuracy_and_f1():
    comparison = pd.DataFrame(
        [
            {"model": "model", "accuracy": 0.75, "macro_f1": 0.6},
            {"model": "baseline", "accuracy": 0.5, "macro_f1": 0.4},
        ]
    )

    chart_data = build_metric_score_chart(comparison)

    assert set(chart_data.columns) == {"model", "metric", "score"}
    assert len(chart_data) == 4


def test_build_prediction_signal_counts_groups_by_ticker_and_signal():
    predictions = pd.DataFrame(
        [
            {"ticker": "AAPL", "predicted_label": 1},
            {"ticker": "AAPL", "predicted_label": -1},
            {"ticker": "MSFT", "predicted_label": 1},
        ]
    )

    counts = build_prediction_signal_counts(predictions)

    assert counts.to_dict("records") == [
        {"ticker": "AAPL", "signal": "Buy", "count": 1},
        {"ticker": "AAPL", "signal": "Sell", "count": 1},
        {"ticker": "MSFT", "signal": "Buy", "count": 1},
    ]


def test_build_weekly_signal_timeline_aggregates_by_ticker_week():
    predictions = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2024-10-01", "predicted_label": 1},
            {"ticker": "AAPL", "date": "2024-10-03", "predicted_label": -1},
            {"ticker": "AAPL", "date": "2024-10-08", "predicted_label": -1},
            {"ticker": "MSFT", "date": "2024-10-02", "predicted_label": 1},
        ]
    )

    timeline = build_weekly_signal_timeline(predictions)

    assert timeline.to_dict("records") == [
        {
            "ticker": "AAPL",
            "week_start": "2024-09-30",
            "weekly_signal": "Hold",
            "signal_score": 0.0,
            "prediction_count": 2,
        },
        {
            "ticker": "MSFT",
            "week_start": "2024-09-30",
            "weekly_signal": "Buy",
            "signal_score": 1.0,
            "prediction_count": 1,
        },
        {
            "ticker": "AAPL",
            "week_start": "2024-10-07",
            "weekly_signal": "Sell",
            "signal_score": -1.0,
            "prediction_count": 1,
        },
    ]


def test_build_weekly_signal_timeline_can_include_empty_weeks():
    predictions = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2024-09-30", "predicted_label": 1},
            {"ticker": "AAPL", "date": "2024-10-14", "predicted_label": -1},
        ]
    )

    timeline = build_weekly_signal_timeline(predictions, include_empty_weeks=True)

    assert timeline[["ticker", "week_start", "weekly_signal", "prediction_count"]].to_dict(
        "records"
    ) == [
        {
            "ticker": "AAPL",
            "week_start": "2024-09-30",
            "weekly_signal": "Buy",
            "prediction_count": 1,
        },
        {
            "ticker": "AAPL",
            "week_start": "2024-10-07",
            "weekly_signal": "No signal",
            "prediction_count": 0,
        },
        {
            "ticker": "AAPL",
            "week_start": "2024-10-14",
            "weekly_signal": "Sell",
            "prediction_count": 1,
        },
    ]


def test_build_return_chart_compares_strategy_to_buy_hold():
    summary = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy_return": 0.05,
                "buy_hold_return": 0.02,
                "excess_return": 0.03,
            }
        ]
    )

    chart_data = build_return_chart(summary)

    assert chart_data.to_dict("records") == [
        {"ticker": "AAPL", "series": "strategy_return", "return": 0.05},
        {"ticker": "AAPL", "series": "buy_hold_return", "return": 0.02},
    ]


def test_build_average_equity_curve_compares_average_equity_lines():
    equity_curve = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "AAPL", "equity": 100, "buy_hold_equity": 100},
            {"date": "2024-01-02", "ticker": "MSFT", "equity": 110, "buy_hold_equity": 120},
        ]
    )

    chart_data = build_average_equity_curve(equity_curve)

    assert chart_data.to_dict("records") == [
        {"date": "2024-01-02", "series": "equity", "equity": 105.0},
        {"date": "2024-01-02", "series": "buy_hold_equity", "equity": 110.0},
    ]
