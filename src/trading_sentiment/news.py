from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = {"ticker", "published_date", "title"}


def load_news_csv(path: str | Path) -> pd.DataFrame:
    """Load news articles from CSV and normalize the expected columns.

    Required columns:
    - ticker
    - published_date
    - title

    Optional columns:
    - summary
    - source
    """
    news = pd.read_csv(path)
    missing = _REQUIRED_COLUMNS - set(news.columns)
    if missing:
        raise ValueError(f"News CSV missing required columns: {sorted(missing)}")

    news = news.copy()
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news["published_date"] = pd.to_datetime(news["published_date"]).dt.date
    news["title"] = news["title"].fillna("").astype(str)

    if "summary" not in news.columns:
        news["summary"] = ""
    else:
        news["summary"] = news["summary"].fillna("").astype(str)

    if "source" not in news.columns:
        news["source"] = "unknown"
    else:
        news["source"] = news["source"].fillna("unknown").astype(str)

    return news[["ticker", "published_date", "title", "summary", "source"]]


def aggregate_daily_news(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple articles into one ticker/date row."""
    if news.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "title", "summary", "source", "article_count"]
        )

    grouped = (
        news.groupby(["ticker", "published_date"], as_index=False)
        .agg(
            title=("title", lambda values: " | ".join(values)),
            summary=("summary", lambda values: " | ".join(values)),
            source=("source", lambda values: " | ".join(sorted(set(values)))),
            article_count=("title", "size"),
        )
        .rename(columns={"published_date": "date"})
    )
    return grouped
