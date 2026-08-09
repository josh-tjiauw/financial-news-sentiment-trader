from pathlib import Path

from trading_sentiment.news import (
    _extract_alpha_vantage_news_items,
    _extract_yahoo_news_item,
    aggregate_daily_news,
    load_news_csv,
)


def test_load_news_csv_normalizes_columns(tmp_path: Path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "ticker,published_date,title\n"
        "aapl,2024-01-02,Apple rises\n",
        encoding="utf-8",
    )

    news = load_news_csv(news_csv)

    assert list(news.columns) == [
        "ticker",
        "published_date",
        "title",
        "summary",
        "source",
        "url",
    ]
    assert news.loc[0, "ticker"] == "AAPL"
    assert news.loc[0, "summary"] == ""
    assert news.loc[0, "source"] == "unknown"
    assert news.loc[0, "url"] == ""


def test_aggregate_daily_news_combines_articles(tmp_path: Path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "ticker,published_date,title,summary,source,url\n"
        "AAPL,2024-01-02,Title one,Summary one,Yahoo,https://example.com/1\n"
        "AAPL,2024-01-02,Title two,Summary two,Reuters,https://example.com/2\n",
        encoding="utf-8",
    )

    daily = aggregate_daily_news(load_news_csv(news_csv))

    assert len(daily) == 1
    assert daily.loc[0, "article_count"] == 2
    assert daily.loc[0, "title"] == "Title one | Title two"
    assert daily.loc[0, "source"] == "Reuters | Yahoo"
    assert daily.loc[0, "url"] == "https://example.com/1 | https://example.com/2"


def test_extract_yahoo_news_item_normalizes_current_yfinance_shape():
    item = {
        "content": {
            "title": "Apple expands AI features",
            "summary": "New features are coming to iPhone users.",
            "pubDate": "2024-01-02T15:30:00Z",
            "provider": {"displayName": "Yahoo Finance"},
            "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-ai"},
        }
    }

    row = _extract_yahoo_news_item("aapl", item)

    assert row == {
        "ticker": "AAPL",
        "published_date": row["published_date"],
        "title": "Apple expands AI features",
        "summary": "New features are coming to iPhone users.",
        "source": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/apple-ai",
    }
    assert str(row["published_date"]) == "2024-01-02"


def test_extract_yahoo_news_item_normalizes_legacy_yfinance_shape():
    item = {
        "title": "Microsoft cloud demand rises",
        "summary": "Cloud growth remains strong.",
        "publisher": "Reuters",
        "link": "https://example.com/msft-cloud",
        "providerPublishTime": 1704209400,
    }

    row = _extract_yahoo_news_item("msft", item)

    assert row["ticker"] == "MSFT"
    assert row["title"] == "Microsoft cloud demand rises"
    assert row["summary"] == "Cloud growth remains strong."
    assert row["source"] == "Reuters"
    assert row["url"] == "https://example.com/msft-cloud"


def test_extract_alpha_vantage_news_items_normalizes_matching_tickers():
    payload = {
        "feed": [
            {
                "title": "Apple and Microsoft rally after earnings",
                "summary": "Large-cap tech stocks climbed after results.",
                "source": "Reuters",
                "url": "https://example.com/tech-rally",
                "time_published": "20241015T143000",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "ticker_sentiment_score": "0.22"},
                    {"ticker": "MSFT", "ticker_sentiment_score": "0.17"},
                    {"ticker": "SPY", "ticker_sentiment_score": "0.05"},
                ],
            }
        ]
    }

    rows = _extract_alpha_vantage_news_items(["aapl", "msft"], payload)

    assert rows == [
        {
            "ticker": "AAPL",
            "published_date": rows[0]["published_date"],
            "title": "Apple and Microsoft rally after earnings",
            "summary": "Large-cap tech stocks climbed after results.",
            "source": "Reuters",
            "url": "https://example.com/tech-rally",
        },
        {
            "ticker": "MSFT",
            "published_date": rows[1]["published_date"],
            "title": "Apple and Microsoft rally after earnings",
            "summary": "Large-cap tech stocks climbed after results.",
            "source": "Reuters",
            "url": "https://example.com/tech-rally",
        },
    ]
    assert str(rows[0]["published_date"]) == "2024-10-15"
