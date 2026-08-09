import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

_REQUIRED_COLUMNS = {"ticker", "published_date", "title"}
_NEWS_COLUMNS = ["ticker", "published_date", "title", "summary", "source", "url"]
_ALPHA_VANTAGE_NEWS_URL = "https://www.alphavantage.co/query"


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


def _coerce_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _format_alpha_vantage_time(value: str | date, boundary: time) -> str:
    return datetime.combine(_coerce_date(value), boundary).strftime("%Y%m%dT%H%M")


def _parse_alpha_vantage_timestamp(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for date_format in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue
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


def _extract_alpha_vantage_news_items(
    tickers: list[str],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize Alpha Vantage News & Sentiment API response rows."""
    requested_tickers = {ticker.upper().strip() for ticker in tickers if ticker.strip()}
    rows: list[dict[str, Any]] = []

    for item in payload.get("feed", []):
        title = item.get("title")
        published_date = _parse_alpha_vantage_timestamp(item.get("time_published"))
        if not title or published_date is None:
            continue

        article_tickers = {
            str(sentiment.get("ticker", "")).upper().strip()
            for sentiment in item.get("ticker_sentiment", [])
            if sentiment.get("ticker")
        }
        matching_tickers = sorted(article_tickers & requested_tickers)
        if not matching_tickers and len(requested_tickers) == 1:
            matching_tickers = sorted(requested_tickers)

        for ticker in matching_tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "published_date": published_date,
                    "title": str(title),
                    "summary": str(item.get("summary") or ""),
                    "source": str(item.get("source") or "Alpha Vantage"),
                    "url": str(item.get("url") or ""),
                }
            )

    return rows


def fetch_alpha_vantage_news(
    tickers: list[str],
    api_key: str,
    start_date: str | date = "2024-10-01",
    end_date: str | date = "2024-12-31",
    limit: int = 1000,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Fetch historical ticker news from Alpha Vantage News & Sentiment API."""
    normalized_tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    if not normalized_tickers:
        return pd.DataFrame(columns=_NEWS_COLUMNS)
    if not api_key:
        raise ValueError("Alpha Vantage API key is required")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    query = urlencode(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(normalized_tickers),
            "time_from": _format_alpha_vantage_time(start_date, time.min),
            "time_to": _format_alpha_vantage_time(end_date, time.max),
            "limit": min(limit, 1000),
            "apikey": api_key,
        }
    )

    with urlopen(f"{_ALPHA_VANTAGE_NEWS_URL}?{query}", timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error: {payload['Error Message']}")
    if "Information" in payload and "feed" not in payload:
        raise ValueError(f"Alpha Vantage response did not include news feed: {payload['Information']}")

    rows = _extract_alpha_vantage_news_items(normalized_tickers, payload)
    return pd.DataFrame(rows, columns=_NEWS_COLUMNS)


def save_news_csv(news: pd.DataFrame, path: str | Path) -> None:
    """Persist normalized news rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output_path, index=False)
