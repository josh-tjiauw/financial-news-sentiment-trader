from pathlib import Path

import pandas as pd

from trading_sentiment.labeling import future_return_label
from trading_sentiment.news import aggregate_daily_news, load_news_csv
from trading_sentiment.prices import load_prices_csv
from trading_sentiment.text import clean_text, combine_title_summary

_DATASET_COLUMNS = [
    "ticker",
    "date",
    "title",
    "summary",
    "source",
    "article_count",
    "combined_text",
    "cleaned_text",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "future_close",
    "future_return",
    "label",
]


def build_dataset(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int = 1,
    neutral_threshold: float = 0.0025,
) -> pd.DataFrame:
    """Join daily news with prices and label each row using future price movement."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    daily_news = aggregate_daily_news(news)
    if daily_news.empty:
        return pd.DataFrame(columns=_DATASET_COLUMNS)

    prices = prices.copy().sort_values(["ticker", "date"])
    prices["future_close"] = prices.groupby("ticker")["close"].shift(-horizon_days)

    dataset = daily_news.merge(prices, on=["ticker", "date"], how="inner")
    dataset = dataset.dropna(subset=["future_close"]).copy()
    if dataset.empty:
        return pd.DataFrame(columns=_DATASET_COLUMNS)

    dataset["future_return"] = (dataset["future_close"] - dataset["close"]) / dataset["close"]
    dataset["label"] = dataset.apply(
        lambda row: int(
            future_return_label(row["close"], row["future_close"], neutral_threshold)
        ),
        axis=1,
    )
    dataset["combined_text"] = dataset.apply(
        lambda row: combine_title_summary(row["title"], row["summary"]), axis=1
    )
    dataset["cleaned_text"] = dataset["combined_text"].apply(clean_text)

    return dataset[_DATASET_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def build_dataset_from_csv(
    news_csv: str | Path,
    prices_csv: str | Path,
    output_csv: str | Path,
    horizon_days: int = 1,
    neutral_threshold: float = 0.0025,
) -> pd.DataFrame:
    """Load raw CSV inputs, build a modeling dataset, and write it to disk."""
    news = load_news_csv(news_csv)
    prices = load_prices_csv(prices_csv)
    dataset = build_dataset(news, prices, horizon_days, neutral_threshold)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset
