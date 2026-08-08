import pandas as pd
import pytest

from trading_sentiment.model import (
    chronological_train_test_split,
    evaluate_naive_baselines,
    train_baseline_model,
)


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": "2024-01-02",
                "cleaned_text": "apple services growth demand strong",
                "close": 100,
                "future_close": 102,
                "future_return": 0.02,
                "label": 1,
            },
            {
                "ticker": "MSFT",
                "date": "2024-01-02",
                "cleaned_text": "microsoft cloud growth resilient demand",
                "close": 200,
                "future_close": 205,
                "future_return": 0.025,
                "label": 1,
            },
            {
                "ticker": "AAPL",
                "date": "2024-01-03",
                "cleaned_text": "apple hardware demand weak pressure",
                "close": 102,
                "future_close": 99,
                "future_return": -0.029,
                "label": -1,
            },
            {
                "ticker": "MSFT",
                "date": "2024-01-03",
                "cleaned_text": "microsoft valuation pressure cautious investors",
                "close": 205,
                "future_close": 202,
                "future_return": -0.015,
                "label": -1,
            },
            {
                "ticker": "NVDA",
                "date": "2024-01-04",
                "cleaned_text": "nvidia ai chip demand expands",
                "close": 300,
                "future_close": 315,
                "future_return": 0.05,
                "label": 1,
            },
            {
                "ticker": "NVDA",
                "date": "2024-01-05",
                "cleaned_text": "nvidia export restrictions weigh shares",
                "close": 315,
                "future_close": 305,
                "future_return": -0.032,
                "label": -1,
            },
        ]
    )


def test_chronological_train_test_split_preserves_future_holdout():
    dataset = _sample_dataset()

    train, test = chronological_train_test_split(dataset, test_size=0.33)

    assert len(train) == 4
    assert len(test) == 2
    assert train["date"].max() <= test["date"].min()


def test_train_baseline_model_returns_metrics_and_predictions():
    result = train_baseline_model(_sample_dataset(), test_size=0.33, max_features=50)

    assert result.metrics["model"] == "tfidf_logistic_regression"
    assert result.metrics["train_row_count"] == 4
    assert result.metrics["test_row_count"] == 2
    assert set(result.metrics["naive_baselines"]) == {
        "majority_class",
        "stratified_random",
        "ticker_prior",
    }
    assert set(result.metrics["naive_baselines"]["majority_class"]) >= {
        "accuracy",
        "macro_f1",
        "predicted_label",
    }
    assert set(result.predictions.columns) >= {
        "ticker",
        "date",
        "close",
        "future_close",
        "future_return",
        "label",
        "predicted_label",
    }


def test_evaluate_naive_baselines_scores_simple_reference_models():
    train = pd.DataFrame(
        [
            {"ticker": "AAPL", "cleaned_text": "strong growth", "label": 1},
            {"ticker": "AAPL", "cleaned_text": "strong demand", "label": 1},
            {"ticker": "MSFT", "cleaned_text": "weak demand", "label": -1},
        ]
    )
    test = pd.DataFrame(
        [
            {"ticker": "AAPL", "cleaned_text": "growth continues", "label": 1},
            {"ticker": "MSFT", "cleaned_text": "pressure remains", "label": -1},
        ]
    )

    metrics = evaluate_naive_baselines(train, test, labels=[-1, 1])

    assert metrics["majority_class"]["predicted_label"] == 1
    assert metrics["majority_class"]["accuracy"] == 0.5
    assert 0 <= metrics["stratified_random"]["macro_f1"] <= 1
    assert metrics["ticker_prior"]["accuracy"] == 1.0


def test_train_baseline_model_requires_two_label_classes():
    dataset = _sample_dataset()
    dataset["label"] = 1

    with pytest.raises(ValueError, match="at least two label classes"):
        train_baseline_model(dataset)
