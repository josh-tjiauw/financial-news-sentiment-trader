from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Artifacts produced by baseline model training."""

    model: Pipeline
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def _validate_modeling_dataset(dataset: pd.DataFrame) -> None:
    required_columns = {"date", "ticker", "cleaned_text", "label"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required columns: {missing}")

    if dataset.empty:
        raise ValueError("dataset must contain at least one row")

    if dataset["label"].nunique() < 2:
        raise ValueError("dataset must contain at least two label classes for logistic regression")


def chronological_train_test_split(
    dataset: pd.DataFrame,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by date order so evaluation is closer to future prediction."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    ordered = dataset.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["date", "ticker"]).reset_index(drop=True)

    test_count = max(1, round(len(ordered) * test_size))
    train_count = len(ordered) - test_count
    if train_count < 1:
        raise ValueError("dataset is too small to create train and test splits")

    train = ordered.iloc[:train_count].copy()
    test = ordered.iloc[train_count:].copy()

    if train["label"].nunique() < 2:
        raise ValueError(
            "training split must contain at least two label classes; add more historical rows "
            "or reduce --test-size"
        )

    return train, test


def build_baseline_pipeline(max_features: int = 5000) -> Pipeline:
    """Create a TF-IDF + logistic regression baseline classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline_model(
    dataset: pd.DataFrame,
    test_size: float = 0.25,
    max_features: int = 5000,
) -> BaselineTrainingResult:
    """Train and evaluate the first NLP baseline model."""
    _validate_modeling_dataset(dataset)
    train, test = chronological_train_test_split(dataset, test_size=test_size)

    model = build_baseline_pipeline(max_features=max_features)
    model.fit(train["cleaned_text"], train["label"])

    predicted_labels = model.predict(test["cleaned_text"])
    prediction_scores = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        prediction_scores = model.predict_proba(test["cleaned_text"]).max(axis=1)

    prediction_columns = [
        column
        for column in [
            "ticker",
            "date",
            "cleaned_text",
            "close",
            "future_close",
            "future_return",
            "label",
        ]
        if column in test.columns
    ]
    predictions = test[prediction_columns].copy()
    predictions["predicted_label"] = predicted_labels
    if prediction_scores is not None:
        predictions["prediction_confidence"] = prediction_scores

    labels = sorted(dataset["label"].unique().tolist())
    metrics: dict[str, Any] = {
        "model": "tfidf_logistic_regression",
        "row_count": int(len(dataset)),
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "test_size": test_size,
        "max_features": max_features,
        "labels": [int(label) for label in labels],
        "accuracy": float(accuracy_score(test["label"], predicted_labels)),
        "macro_f1": float(
            f1_score(test["label"], predicted_labels, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            test["label"],
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    return BaselineTrainingResult(model=model, metrics=metrics, predictions=predictions)


def train_baseline_from_csv(
    dataset_csv: str | Path,
    metrics_output: str | Path,
    predictions_output: str | Path | None = None,
    model_output: str | Path | None = None,
    test_size: float = 0.25,
    max_features: int = 5000,
) -> BaselineTrainingResult:
    """Train the baseline model from a processed dataset and write artifacts."""
    dataset = pd.read_csv(dataset_csv)
    result = train_baseline_model(dataset, test_size=test_size, max_features=max_features)

    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    if predictions_output is not None:
        predictions_path = Path(predictions_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(predictions_path, index=False)

    if model_output is not None:
        import joblib

        model_path = Path(model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, model_path)

    return result
