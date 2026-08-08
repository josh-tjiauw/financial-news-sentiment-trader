from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd

from trading_sentiment.news import save_news_csv

_NORMALIZED_NEWS_COLUMNS = ["ticker", "published_date", "title", "summary", "source", "url"]


def _require_column(data: pd.DataFrame, column: str, purpose: str) -> None:
    if column not in data.columns:
        raise ValueError(f"Input CSV missing {purpose} column: {column}")


def _normalize_tickers(tickers: list[str] | None) -> set[str] | None:
    if not tickers:
        return None
    return {ticker.upper().strip() for ticker in tickers if ticker.strip()}


def _fetch_url_range(url: str, start_byte: int, end_byte: int, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _csv_header_from_bytes(content: bytes) -> list[str]:
    first_line = content.splitlines()[0].decode("utf-8-sig", errors="replace")
    return next(csv.reader([first_line]))


def _candidate_rows_from_csv_bytes(
    content: bytes,
    header: list[str],
    skip_header: bool = False,
) -> pd.DataFrame:
    text = content.decode("utf-8", errors="replace")
    if skip_header:
        _, _, text = text.partition("\n")
    else:
        _, _, text = text.partition("\n")

    reader = csv.reader(io.StringIO(text))
    expected_columns = len(header)
    rows = [row for row in reader if len(row) == expected_columns]
    if not rows:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(rows, columns=header)


def _normalize_news_frame(
    raw: pd.DataFrame,
    ticker_column: str,
    date_column: str,
    title_column: str,
    summary_column: str | None = None,
    source_column: str | None = None,
    url_column: str | None = None,
    source_name: str = "imported",
    tickers: set[str] | None = None,
) -> pd.DataFrame:
    _require_column(raw, ticker_column, "ticker")
    _require_column(raw, date_column, "date")
    _require_column(raw, title_column, "title")

    optional_columns = [summary_column, source_column, url_column]
    for column in optional_columns:
        if column:
            _require_column(raw, column, "optional mapped")

    normalized = pd.DataFrame()
    normalized["ticker"] = raw[ticker_column].fillna("").astype(str).str.upper().str.strip()
    normalized["published_date"] = pd.to_datetime(raw[date_column], errors="coerce").dt.date
    normalized["title"] = raw[title_column].fillna("").astype(str).str.strip()

    if summary_column:
        normalized["summary"] = raw[summary_column].fillna("").astype(str).str.strip()
    else:
        normalized["summary"] = ""

    if source_column:
        normalized["source"] = raw[source_column].fillna(source_name).astype(str).str.strip()
    else:
        normalized["source"] = source_name

    if url_column:
        normalized["url"] = raw[url_column].fillna("").astype(str).str.strip()
    else:
        normalized["url"] = ""

    normalized = normalized.dropna(subset=["published_date"])
    normalized = normalized[(normalized["ticker"] != "") & (normalized["title"] != "")]

    if tickers:
        normalized = normalized[normalized["ticker"].isin(tickers)]

    return normalized[_NORMALIZED_NEWS_COLUMNS]


def import_news_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    ticker_column: str,
    date_column: str,
    title_column: str,
    summary_column: str | None = None,
    source_column: str | None = None,
    url_column: str | None = None,
    source_name: str = "imported",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize an arbitrary historical-news CSV into the project news schema.

    This lets the project ingest Kaggle/manual/vendor exports without hard-coding one provider.
    Required normalized output columns:
    ticker,published_date,title,summary,source,url
    """
    raw = pd.read_csv(input_csv)
    normalized = _normalize_news_frame(
        raw=raw,
        ticker_column=ticker_column,
        date_column=date_column,
        title_column=title_column,
        summary_column=summary_column,
        source_column=source_column,
        url_column=url_column,
        source_name=source_name,
        tickers=_normalize_tickers(tickers),
    )

    normalized = normalized[_NORMALIZED_NEWS_COLUMNS].sort_values(
        ["ticker", "published_date", "title"]
    )
    normalized = normalized.drop_duplicates().reset_index(drop=True)

    save_news_csv(normalized, output_csv)
    return normalized


def prepare_historical_news_csv(
    input_csv_or_url: str | Path,
    output_csv: str | Path,
    ticker_column: str,
    date_column: str,
    title_column: str,
    summary_column: str | None = None,
    source_column: str | None = None,
    url_column: str | None = None,
    source_name: str = "historical",
    tickers: list[str] | None = None,
    chunk_size: int = 100_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Stream, filter, and normalize a large local/remote historical news CSV.

    This is the safer path for large datasets such as FNSPID/Benzinga because it can read
    in chunks and keep only the ticker subset needed for the project.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be greater than zero when provided")

    wanted_tickers = _normalize_tickers(tickers)
    normalized_chunks: list[pd.DataFrame] = []
    kept_rows = 0

    chunks: Iterable[pd.DataFrame] = pd.read_csv(input_csv_or_url, chunksize=chunk_size)
    for chunk in chunks:
        normalized = _normalize_news_frame(
            raw=chunk,
            ticker_column=ticker_column,
            date_column=date_column,
            title_column=title_column,
            summary_column=summary_column,
            source_column=source_column,
            url_column=url_column,
            source_name=source_name,
            tickers=wanted_tickers,
        )

        if normalized.empty:
            continue

        if max_rows is not None:
            remaining_rows = max_rows - kept_rows
            if remaining_rows <= 0:
                break
            normalized = normalized.head(remaining_rows)

        normalized_chunks.append(normalized)
        kept_rows += len(normalized)

        if max_rows is not None and kept_rows >= max_rows:
            break

    if normalized_chunks:
        prepared = pd.concat(normalized_chunks, ignore_index=True)
        prepared = prepared.sort_values(["ticker", "published_date", "title"])
        prepared = prepared.drop_duplicates().reset_index(drop=True)
    else:
        prepared = pd.DataFrame(columns=_NORMALIZED_NEWS_COLUMNS)

    save_news_csv(prepared, output_csv)
    return prepared


def _combine_unique_news_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame(columns=_NORMALIZED_NEWS_COLUMNS)
    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values(["ticker", "published_date", "title"])
    return combined.drop_duplicates().reset_index(drop=True)


def _ticker_counts(data: pd.DataFrame, tickers: set[str]) -> dict[str, int]:
    if data.empty:
        return {ticker: 0 for ticker in tickers}
    counts = data["ticker"].value_counts().to_dict()
    return {ticker: int(counts.get(ticker, 0)) for ticker in tickers}


def _targeted_ticker_rows(
    normalized: pd.DataFrame,
    current_counts: dict[str, int],
    min_rows_per_ticker: int,
) -> pd.DataFrame:
    targeted_chunks: list[pd.DataFrame] = []
    for ticker, current_count in current_counts.items():
        remaining = min_rows_per_ticker - current_count
        if remaining <= 0:
            continue
        ticker_rows = normalized[normalized["ticker"] == ticker].head(remaining)
        if not ticker_rows.empty:
            targeted_chunks.append(ticker_rows)
    return _combine_unique_news_chunks(targeted_chunks)


def sample_remote_historical_news_csv(
    url: str,
    output_csv: str | Path,
    ticker_column: str,
    date_column: str,
    title_column: str,
    summary_column: str | None = None,
    source_column: str | None = None,
    url_column: str | None = None,
    source_name: str = "remote-sample",
    tickers: list[str] | None = None,
    range_bytes: int = 25_000_000,
    step_bytes: int | None = None,
    max_ranges: int = 20,
    max_rows: int = 5_000,
    start_byte: int = 0,
    timeout_seconds: int = 120,
    min_rows_per_ticker: int | None = None,
) -> pd.DataFrame:
    """Sample a huge remote CSV with HTTP byte ranges and normalize matching rows.

    This is designed for very large direct-download files where normal CSV streaming is too slow.
    It samples bounded byte ranges, keeps complete parseable CSV rows, filters tickers, and writes a
    normalized subset. With ``min_rows_per_ticker``, it keeps scanning ranges until every requested
    ticker has enough unique rows or ``max_ranges`` is exhausted. Because byte ranges can begin/end
    inside quoted article text, this is a sampling tool rather than a guaranteed full extractor.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("url must be an HTTP(S) URL")
    if range_bytes < 1:
        raise ValueError("range_bytes must be greater than zero")
    if step_bytes is not None and step_bytes < 1:
        raise ValueError("step_bytes must be greater than zero when provided")
    if max_ranges < 1:
        raise ValueError("max_ranges must be greater than zero")
    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero")
    if start_byte < 0:
        raise ValueError("start_byte cannot be negative")
    if min_rows_per_ticker is not None and min_rows_per_ticker < 1:
        raise ValueError("min_rows_per_ticker must be greater than zero when provided")

    actual_step_bytes = step_bytes or range_bytes
    wanted_tickers = _normalize_tickers(tickers)
    if min_rows_per_ticker is not None:
        if not wanted_tickers:
            raise ValueError("min_rows_per_ticker requires a ticker allowlist")
        minimum_total_rows = len(wanted_tickers) * min_rows_per_ticker
        if max_rows < minimum_total_rows:
            raise ValueError("max_rows must be at least tickers * min_rows_per_ticker")

    normalized_chunks: list[pd.DataFrame] = []
    kept_rows = 0

    first_content = _fetch_url_range(
        url=url,
        start_byte=0,
        end_byte=min(range_bytes - 1, 250_000),
        timeout_seconds=timeout_seconds,
    )
    header = _csv_header_from_bytes(first_content)

    for range_index in range(max_ranges):
        current_start = start_byte + (range_index * actual_step_bytes)
        current_end = current_start + range_bytes - 1
        content = _fetch_url_range(
            url=url,
            start_byte=current_start,
            end_byte=current_end,
            timeout_seconds=timeout_seconds,
        )
        if not content:
            break

        raw = _candidate_rows_from_csv_bytes(
            content=content,
            header=header,
            skip_header=current_start == 0,
        )
        if raw.empty:
            continue

        normalized = _normalize_news_frame(
            raw=raw,
            ticker_column=ticker_column,
            date_column=date_column,
            title_column=title_column,
            summary_column=summary_column,
            source_column=source_column,
            url_column=url_column,
            source_name=source_name,
            tickers=wanted_tickers,
        )
        if normalized.empty:
            continue

        if min_rows_per_ticker is not None and wanted_tickers is not None:
            current_counts = _ticker_counts(
                _combine_unique_news_chunks(normalized_chunks),
                wanted_tickers,
            )
            normalized = _targeted_ticker_rows(
                normalized=normalized,
                current_counts=current_counts,
                min_rows_per_ticker=min_rows_per_ticker,
            )
            if normalized.empty:
                continue
        else:
            remaining_rows = max_rows - kept_rows
            if remaining_rows <= 0:
                break
            normalized = normalized.head(remaining_rows)

        normalized_chunks.append(normalized)
        sampled_so_far = _combine_unique_news_chunks(normalized_chunks)
        kept_rows = len(sampled_so_far)

        if min_rows_per_ticker is not None and wanted_tickers is not None:
            current_counts = _ticker_counts(sampled_so_far, wanted_tickers)
            if all(count >= min_rows_per_ticker for count in current_counts.values()):
                break
        elif kept_rows >= max_rows:
            break

    sampled = _combine_unique_news_chunks(normalized_chunks).head(max_rows)

    save_news_csv(sampled, output_csv)
    return sampled
