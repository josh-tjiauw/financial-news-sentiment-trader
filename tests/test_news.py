from pathlib import Path

from trading_sentiment.news import aggregate_daily_news, load_news_csv


def test_load_news_csv_normalizes_columns(tmp_path: Path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "ticker,published_date,title\n"
        "aapl,2024-01-02,Apple rises\n",
        encoding="utf-8",
    )

    news = load_news_csv(news_csv)

    assert list(news.columns) == ["ticker", "published_date", "title", "summary", "source"]
    assert news.loc[0, "ticker"] == "AAPL"
    assert news.loc[0, "summary"] == ""
    assert news.loc[0, "source"] == "unknown"


def test_aggregate_daily_news_combines_articles(tmp_path: Path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "ticker,published_date,title,summary,source\n"
        "AAPL,2024-01-02,Title one,Summary one,Yahoo\n"
        "AAPL,2024-01-02,Title two,Summary two,Reuters\n",
        encoding="utf-8",
    )

    daily = aggregate_daily_news(load_news_csv(news_csv))

    assert len(daily) == 1
    assert daily.loc[0, "article_count"] == 2
    assert daily.loc[0, "title"] == "Title one | Title two"
    assert daily.loc[0, "source"] == "Reuters | Yahoo"
