from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_REQUIRED_COLUMNS = {"ticker", "published_date", "title"}
_NEWS_COLUMNS = ["ticker", "published_date", "title", "summary", "source", "url"]


def load_news_csv(path: str | Path) -> pd.DataFrame:
    """Load news articles from CSV and normalize the expected columns.

    Required columns:
    - ticker
    - published_date
    - title

    Optional columns:
    - summary
    - source
    - url
    """
    news = pd.read_csv(path)
    missing = _REQUIRED_COLUMNS - set(news.columns)
    if missing:
        raise ValueError(f"News CSV missing required columns: {sorted(missing)}")

    news = news.copy()
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news["published_date"] = pd.to_datetime(news["published_date"]).dt.date
    news["title"] = news["title"].fillna("").astype(str)

    for optional_column, default in {
        "summary": "",
        "source": "unknown",
        "url": "",
    }.items():
        if optional_column not in news.columns:
            news[optional_column] = default
        else:
            news[optional_column] = news[optional_column].fillna(default).astype(str)

    return news[_NEWS_COLUMNS]


def aggregate_daily_news(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple articles into one ticker/date row."""
    if news.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "title", "summary", "source", "url", "article_count"]
        )

    news = news.copy()
    if "url" not in news.columns:
        news["url"] = ""

    grouped = (
        news.groupby(["ticker", "published_date"], as_index=False)
        .agg(
            title=("title", lambda values: " | ".join(values)),
            summary=("summary", lambda values: " | ".join(values)),
            source=("source", lambda values: " | ".join(sorted(set(values)))),
            url=("url", lambda values: " | ".join(value for value in values if value)),
            article_count=("title", "size"),
        )
        .rename(columns={"published_date": "date"})
    )
    return grouped


def _parse_yahoo_timestamp(value: Any) -> datetime | None:
    """Parse known Yahoo/yfinance timestamp fields into a timezone-aware datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None

    return None


def _extract_yahoo_news_item(ticker: str, item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one yfinance Yahoo news item."""
    content = item.get("content", item)
    title = content.get("title") or item.get("title")
    if not title:
        return None

    summary = content.get("summary") or item.get("summary") or ""
    provider = content.get("provider") or item.get("publisher") or {}
    source = provider.get("displayName") if isinstance(provider, dict) else provider
    source = source or "Yahoo Finance"

    canonical_url = content.get("canonicalUrl") or item.get("link") or {}
    if isinstance(canonical_url, dict):
        url = canonical_url.get("url", "")
    else:
        url = canonical_url or ""

    published = (
        _parse_yahoo_timestamp(content.get("pubDate"))
        or _parse_yahoo_timestamp(content.get("displayTime"))
        or _parse_yahoo_timestamp(item.get("providerPublishTime"))
    )
    if published is None:
        published = datetime.now(tz=UTC)

    return {
        "ticker": ticker.upper(),
        "published_date": published.date(),
        "title": str(title),
        "summary": str(summary),
        "source": str(source),
        "url": str(url),
    }


def fetch_yahoo_news(tickers: list[str], max_articles_per_ticker: int = 25) -> pd.DataFrame:
    """Fetch recent Yahoo Finance news through yfinance and return normalized rows.

    Yahoo only exposes recent news through this endpoint. For historical news, the project will need
    a paid/news-archive provider later.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install project dependencies with `pip install -e .[dev]` first") from exc

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        normalized_ticker = ticker.upper().strip()
        if not normalized_ticker:
            continue

        ticker_news = yf.Ticker(normalized_ticker).news or []
        for item in ticker_news[:max_articles_per_ticker]:
            row = _extract_yahoo_news_item(normalized_ticker, item)
            if row:
                rows.append(row)

    return pd.DataFrame(rows, columns=_NEWS_COLUMNS)


def save_news_csv(news: pd.DataFrame, path: str | Path) -> None:
    """Persist normalized news rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output_path, index=False)
