from pathlib import Path

import pytest

import trading_sentiment.importers as importers
from trading_sentiment.importers import (
    import_news_csv,
    prepare_historical_news_csv,
    sample_remote_historical_news_csv,
)
from trading_sentiment.news import load_news_csv


def test_import_news_csv_maps_arbitrary_columns(tmp_path: Path):
    input_csv = tmp_path / "historical_news.csv"
    output_csv = tmp_path / "normalized_news.csv"
    input_csv.write_text(
        "symbol,date,headline,body,provider,link\n"
        "aapl,2024-01-02,Apple rises,Services demand improves,Reuters,https://example.com/aapl\n"
        "msft,2024-01-03,Microsoft cloud grows,Cloud remains strong,Bloomberg,https://example.com/msft\n",
        encoding="utf-8",
    )

    imported = import_news_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        ticker_column="symbol",
        date_column="date",
        title_column="headline",
        summary_column="body",
        source_column="provider",
        url_column="link",
    )

    assert len(imported) == 2
    assert imported.loc[0, "ticker"] == "AAPL"
    assert imported.loc[0, "published_date"].isoformat() == "2024-01-02"
    assert imported.loc[0, "title"] == "Apple rises"
    assert imported.loc[0, "summary"] == "Services demand improves"
    assert output_csv.exists()

    loaded = load_news_csv(output_csv)
    assert list(loaded.columns) == ["ticker", "published_date", "title", "summary", "source", "url"]


def test_import_news_csv_filters_tickers_and_drops_bad_rows(tmp_path: Path):
    input_csv = tmp_path / "historical_news.csv"
    output_csv = tmp_path / "normalized_news.csv"
    input_csv.write_text(
        "ticker,published,headline\n"
        "AAPL,2024-01-02,Apple rises\n"
        "MSFT,not-a-date,Microsoft invalid date\n"
        "NVDA,2024-01-03,Nvidia rises\n"
        ",2024-01-04,Missing ticker\n"
        "TSLA,2024-01-05,\n",
        encoding="utf-8",
    )

    imported = import_news_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        ticker_column="ticker",
        date_column="published",
        title_column="headline",
        source_name="kaggle",
        tickers=["aapl", "nvda"],
    )

    assert imported["ticker"].tolist() == ["AAPL", "NVDA"]
    assert set(imported["source"]) == {"kaggle"}


def test_import_news_csv_requires_mapped_columns(tmp_path: Path):
    input_csv = tmp_path / "historical_news.csv"
    output_csv = tmp_path / "normalized_news.csv"
    input_csv.write_text("ticker,date\nAAPL,2024-01-02\n", encoding="utf-8")

    with pytest.raises(ValueError, match="title"):
        import_news_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            ticker_column="ticker",
            date_column="date",
            title_column="headline",
        )


def test_prepare_historical_news_csv_streams_filters_and_caps_rows(tmp_path: Path):
    input_csv = tmp_path / "large_historical_news.csv"
    output_csv = tmp_path / "prepared_news.csv"
    input_csv.write_text(
        "symbol,published_at,headline,body\n"
        "AAPL,2024-01-01,Apple one,Body one\n"
        "MSFT,2024-01-01,Microsoft one,Body two\n"
        "AAPL,2024-01-02,Apple two,Body three\n"
        "NVDA,2024-01-03,Nvidia one,Body four\n",
        encoding="utf-8",
    )

    prepared = prepare_historical_news_csv(
        input_csv_or_url=input_csv,
        output_csv=output_csv,
        ticker_column="symbol",
        date_column="published_at",
        title_column="headline",
        summary_column="body",
        source_name="FNSPID",
        tickers=["AAPL", "NVDA"],
        chunk_size=2,
        max_rows=2,
    )

    assert len(prepared) == 2
    assert prepared["ticker"].tolist() == ["AAPL", "AAPL"]
    assert set(prepared["source"]) == {"FNSPID"}
    assert output_csv.exists()


def test_prepare_historical_news_csv_validates_chunk_size(tmp_path: Path):
    input_csv = tmp_path / "historical_news.csv"
    output_csv = tmp_path / "prepared_news.csv"
    input_csv.write_text("ticker,date,title\nAAPL,2024-01-02,Apple rises\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chunk_size"):
        prepare_historical_news_csv(
            input_csv_or_url=input_csv,
            output_csv=output_csv,
            ticker_column="ticker",
            date_column="date",
            title_column="title",
            chunk_size=0,
        )


def test_sample_remote_historical_news_csv_uses_range_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_csv = tmp_path / "sampled_remote_news.csv"
    header = "Date,Article_title,Stock_symbol,Url,Publisher,Lsa_summary\n"
    first_range = (
        header
        + "2024-01-01,Apple rises,AAPL,https://example.com/a,Reuters,Apple summary\n"
        + "2024-01-01,Microsoft rises,MSFT,https://example.com/m,Bloomberg,Microsoft summary\n"
    ).encode()
    second_range = (
        "partial row to skip\n"
        + "2024-01-02,Nvidia rises,NVDA,https://example.com/n,CNBC,Nvidia summary\n"
        + "2024-01-03,Apple falls,AAPL,https://example.com/a2,Reuters,Apple down\n"
    ).encode()
    calls: list[tuple[int, int]] = []

    def fake_fetch(url: str, start_byte: int, end_byte: int, timeout_seconds: int) -> bytes:
        calls.append((start_byte, end_byte))
        if len(calls) == 1:
            return first_range
        if start_byte == 0:
            return first_range
        return second_range

    monkeypatch.setattr(importers, "_fetch_url_range", fake_fetch)

    sampled = sample_remote_historical_news_csv(
        url="https://example.com/huge.csv",
        output_csv=output_csv,
        ticker_column="Stock_symbol",
        date_column="Date",
        title_column="Article_title",
        summary_column="Lsa_summary",
        source_column="Publisher",
        url_column="Url",
        source_name="FNSPID",
        tickers=["AAPL", "NVDA"],
        range_bytes=100,
        step_bytes=100,
        max_ranges=2,
        max_rows=10,
    )

    assert sampled["ticker"].tolist() == ["AAPL", "AAPL", "NVDA"]
    assert output_csv.exists()
    assert calls[0] == (0, 99)
    assert calls[-1] == (100, 199)


def test_sample_remote_historical_news_csv_scans_until_ticker_targets_are_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_csv = tmp_path / "targeted_sampled_remote_news.csv"
    header = "Date,Article_title,Stock_symbol,Url,Publisher,Lsa_summary\n"
    header_range = header.encode()
    ranges = {
        0: (header + "2024-01-01,Apple one,AAPL,https://example.com/a1,Reuters,A\n").encode(),
        100: (
            "partial row to skip\n"
            "2024-01-02,Apple two,AAPL,https://example.com/a2,Reuters,A\n"
        ).encode(),
        200: (
            "partial row to skip\n"
            "2024-01-03,Microsoft one,MSFT,https://example.com/m1,Reuters,M\n"
        ).encode(),
        300: (
            "partial row to skip\n"
            "2024-01-04,Microsoft two,MSFT,https://example.com/m2,Reuters,M\n"
        ).encode(),
    }
    calls: list[tuple[int, int]] = []

    def fake_fetch(url: str, start_byte: int, end_byte: int, timeout_seconds: int) -> bytes:
        calls.append((start_byte, end_byte))
        if len(calls) == 1:
            return header_range
        return ranges.get(start_byte, b"")

    monkeypatch.setattr(importers, "_fetch_url_range", fake_fetch)

    sampled = sample_remote_historical_news_csv(
        url="https://example.com/huge.csv",
        output_csv=output_csv,
        ticker_column="Stock_symbol",
        date_column="Date",
        title_column="Article_title",
        summary_column="Lsa_summary",
        source_column="Publisher",
        url_column="Url",
        tickers=["AAPL", "MSFT"],
        range_bytes=100,
        step_bytes=100,
        max_ranges=10,
        max_rows=4,
        min_rows_per_ticker=2,
    )

    assert sampled["ticker"].value_counts().to_dict() == {"AAPL": 2, "MSFT": 2}
    assert output_csv.exists()
    assert calls[-1] == (300, 399)


def test_sample_remote_historical_news_csv_requires_tickers_for_ticker_targets(tmp_path: Path):
    with pytest.raises(ValueError, match="ticker allowlist"):
        sample_remote_historical_news_csv(
            url="https://example.com/huge.csv",
            output_csv=tmp_path / "sampled.csv",
            ticker_column="ticker",
            date_column="date",
            title_column="title",
            min_rows_per_ticker=2,
        )


def test_sample_remote_historical_news_csv_requires_http_url(tmp_path: Path):
    with pytest.raises(ValueError, match="HTTP"):
        sample_remote_historical_news_csv(
            url="downloads/news.csv",
            output_csv=tmp_path / "sampled.csv",
            ticker_column="ticker",
            date_column="date",
            title_column="title",
        )
