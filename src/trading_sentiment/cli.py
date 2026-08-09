import argparse
import os
from pathlib import Path

from trading_sentiment.backtest import backtest_predictions_from_csv
from trading_sentiment.database import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_SCHEMA_PATH,
    initialize_database,
    list_tables,
)
from trading_sentiment.dataset import build_dataset_from_csv
from trading_sentiment.importers import (
    import_news_csv,
    prepare_historical_news_csv,
    sample_remote_historical_news_csv,
)
from trading_sentiment.model import train_baseline_from_csv
from trading_sentiment.news import fetch_alpha_vantage_news, fetch_yahoo_news, save_news_csv
from trading_sentiment.prices import fetch_many_price_histories, save_prices_csv


def _split_tickers(value: str) -> list[str]:
    return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]


def fetch_news_command(args: argparse.Namespace) -> None:
    news = fetch_yahoo_news(
        _split_tickers(args.tickers),
        max_articles_per_ticker=args.max_articles_per_ticker,
    )
    save_news_csv(news, args.output)
    print(f"Saved {len(news)} news rows to {args.output}")


def fetch_alpha_vantage_news_command(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY "
            "or pass --api-key."
        )

    news = fetch_alpha_vantage_news(
        _split_tickers(args.tickers),
        api_key=api_key,
        start_date=args.start,
        end_date=args.end,
        limit=args.limit,
        sort=args.sort,
        request_interval_seconds=args.request_interval_seconds,
    )
    save_news_csv(news, args.output)
    print(
        f"Saved {len(news)} Alpha Vantage news rows for {args.start} to "
        f"{args.end} at {args.output}"
    )


def fetch_prices_command(args: argparse.Namespace) -> None:
    prices = fetch_many_price_histories(_split_tickers(args.tickers), args.start, args.end)
    save_prices_csv(prices, args.output)
    print(f"Saved {len(prices)} price rows to {args.output}")


def import_news_csv_command(args: argparse.Namespace) -> None:
    tickers = _split_tickers(args.tickers) if args.tickers else None
    news = import_news_csv(
        input_csv=args.input,
        output_csv=args.output,
        ticker_column=args.ticker_column,
        date_column=args.date_column,
        title_column=args.title_column,
        summary_column=args.summary_column,
        source_column=args.source_column,
        url_column=args.url_column,
        source_name=args.source_name,
        tickers=tickers,
    )
    print(f"Imported {len(news)} normalized news rows to {args.output}")


def prepare_historical_news_command(args: argparse.Namespace) -> None:
    tickers = _split_tickers(args.tickers) if args.tickers else None
    news = prepare_historical_news_csv(
        input_csv_or_url=args.input,
        output_csv=args.output,
        ticker_column=args.ticker_column,
        date_column=args.date_column,
        title_column=args.title_column,
        summary_column=args.summary_column,
        source_column=args.source_column,
        url_column=args.url_column,
        source_name=args.source_name,
        tickers=tickers,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )
    print(f"Prepared {len(news)} normalized historical news rows to {args.output}")


def sample_remote_news_command(args: argparse.Namespace) -> None:
    tickers = _split_tickers(args.tickers) if args.tickers else None
    news = sample_remote_historical_news_csv(
        url=args.url,
        output_csv=args.output,
        ticker_column=args.ticker_column,
        date_column=args.date_column,
        title_column=args.title_column,
        summary_column=args.summary_column,
        source_column=args.source_column,
        url_column=args.url_column,
        source_name=args.source_name,
        tickers=tickers,
        range_bytes=args.range_bytes,
        step_bytes=args.step_bytes,
        max_ranges=args.max_ranges,
        max_rows=args.max_rows,
        start_byte=args.start_byte,
        timeout_seconds=args.timeout_seconds,
        min_rows_per_ticker=args.min_rows_per_ticker,
    )
    print(f"Sampled {len(news)} normalized remote news rows to {args.output}")


def build_dataset_command(args: argparse.Namespace) -> None:
    dataset = build_dataset_from_csv(
        news_csv=args.news,
        prices_csv=args.prices,
        output_csv=args.output,
        horizon_days=args.horizon_days,
        neutral_threshold=args.neutral_threshold,
    )
    print(f"Saved {len(dataset)} modeling rows to {args.output}")


def train_baseline_command(args: argparse.Namespace) -> None:
    result = train_baseline_from_csv(
        dataset_csv=args.dataset,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        model_output=args.model_output,
        test_size=args.test_size,
        max_features=args.max_features,
    )
    accuracy = result.metrics["accuracy"]
    macro_f1 = result.metrics["macro_f1"]
    print(
        f"Trained TF-IDF + logistic regression baseline: "
        f"accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}"
    )
    print(f"Saved metrics to {args.metrics_output}")
    if args.predictions_output:
        print(f"Saved predictions to {args.predictions_output}")
    if args.model_output:
        print(f"Saved model to {args.model_output}")


def backtest_predictions_command(args: argparse.Namespace) -> None:
    summary, trades, equity_curve = backtest_predictions_from_csv(
        predictions_csv=args.predictions,
        summary_output=args.summary_output,
        trades_output=args.trades_output,
        equity_output=args.equity_output,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        slippage_pct=args.slippage_pct,
    )
    print(f"Backtested {len(summary)} ticker strategies")
    print(f"Saved summary to {args.summary_output}")
    if args.trades_output:
        print(f"Saved {len(trades)} trades to {args.trades_output}")
    if args.equity_output:
        print(f"Saved {len(equity_curve)} equity curve rows to {args.equity_output}")


def init_db_command(args: argparse.Namespace) -> None:
    database_path = initialize_database(database_path=args.database, schema_path=args.schema)
    tables = list_tables(database_path)
    print(f"Initialized local app database at {database_path}")
    print(f"Created/verified {len(tables)} tables: {', '.join(tables)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-sentiment",
        description="Financial news sentiment trading research toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_news = subparsers.add_parser("fetch-news", help="Download recent Yahoo Finance news")
    fetch_news.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA",
    )
    fetch_news.add_argument(
        "--max-articles-per-ticker",
        default=25,
        type=int,
        help="Maximum recent Yahoo Finance articles to keep per ticker",
    )
    fetch_news.add_argument(
        "--output",
        default=Path("data/raw/news.csv"),
        type=Path,
        help="Output CSV path",
    )
    fetch_news.set_defaults(func=fetch_news_command)

    fetch_alpha_news = subparsers.add_parser(
        "fetch-alpha-vantage-news",
        help="Download historical Alpha Vantage market news for a dated project window",
    )
    fetch_alpha_news.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA",
    )
    fetch_alpha_news.add_argument(
        "--start",
        default="2024-09-30",
        help="Start date, YYYY-MM-DD",
    )
    fetch_alpha_news.add_argument(
        "--end",
        default="2024-12-16",
        help="End date, YYYY-MM-DD",
    )
    fetch_alpha_news.add_argument(
        "--sort",
        default="EARLIEST",
        choices=["LATEST", "EARLIEST", "RELEVANCE"],
        help="Alpha Vantage result sort order",
    )
    fetch_alpha_news.add_argument(
        "--limit",
        default=1000,
        type=int,
        help="Maximum Alpha Vantage articles to request, capped at 1000",
    )
    fetch_alpha_news.add_argument(
        "--request-interval-seconds",
        default=1.1,
        type=float,
        help="Delay between per-ticker API requests for free-key rate limits",
    )
    fetch_alpha_news.add_argument(
        "--api-key",
        default=None,
        help="Alpha Vantage API key; defaults to ALPHA_VANTAGE_API_KEY",
    )
    fetch_alpha_news.add_argument(
        "--output",
        default=Path("data/raw/news.csv"),
        type=Path,
        help="Output CSV path",
    )
    fetch_alpha_news.set_defaults(func=fetch_alpha_vantage_news_command)

    fetch_prices = subparsers.add_parser("fetch-prices", help="Download OHLCV prices")
    fetch_prices.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA",
    )
    fetch_prices.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    fetch_prices.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    fetch_prices.add_argument(
        "--output",
        default=Path("data/raw/prices.csv"),
        type=Path,
        help="Output CSV path",
    )
    fetch_prices.set_defaults(func=fetch_prices_command)

    import_news = subparsers.add_parser(
        "import-news-csv",
        help="Normalize a historical/news-provider CSV into the project news schema",
    )
    import_news.add_argument("--input", required=True, type=Path, help="Input news CSV")
    import_news.add_argument(
        "--output",
        default=Path("data/raw/news.csv"),
        type=Path,
        help="Output normalized news CSV",
    )
    import_news.add_argument("--ticker-column", required=True, help="Input column for ticker")
    import_news.add_argument("--date-column", required=True, help="Input column for publish date")
    import_news.add_argument("--title-column", required=True, help="Input column for title/headline")
    import_news.add_argument("--summary-column", default=None, help="Optional summary/body column")
    import_news.add_argument("--source-column", default=None, help="Optional source/provider column")
    import_news.add_argument("--url-column", default=None, help="Optional article URL column")
    import_news.add_argument(
        "--source-name",
        default="imported",
        help="Source value when no source column is provided",
    )
    import_news.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker allowlist, e.g. AAPL,MSFT,NVDA",
    )
    import_news.set_defaults(func=import_news_csv_command)

    prepare_news = subparsers.add_parser(
        "prepare-historical-news",
        help="Stream/filter a large local or remote historical news CSV into project format",
    )
    prepare_news.add_argument(
        "--input",
        required=True,
        help="Input CSV path or HTTPS URL",
    )
    prepare_news.add_argument(
        "--output",
        default=Path("data/raw/news.csv"),
        type=Path,
        help="Output normalized news CSV",
    )
    prepare_news.add_argument("--ticker-column", required=True, help="Input column for ticker")
    prepare_news.add_argument("--date-column", required=True, help="Input column for publish date")
    prepare_news.add_argument("--title-column", required=True, help="Input column for title/headline")
    prepare_news.add_argument("--summary-column", default=None, help="Optional summary/body column")
    prepare_news.add_argument("--source-column", default=None, help="Optional source/provider column")
    prepare_news.add_argument("--url-column", default=None, help="Optional article URL column")
    prepare_news.add_argument(
        "--source-name",
        default="historical",
        help="Source value when no source column is provided",
    )
    prepare_news.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker allowlist, e.g. AAPL,MSFT,NVDA",
    )
    prepare_news.add_argument(
        "--chunk-size",
        default=100_000,
        type=int,
        help="Rows per CSV chunk while streaming large files",
    )
    prepare_news.add_argument(
        "--max-rows",
        default=None,
        type=int,
        help="Optional cap on normalized rows kept after filtering",
    )
    prepare_news.set_defaults(func=prepare_historical_news_command)

    sample_remote_news = subparsers.add_parser(
        "sample-remote-news",
        help="Sample huge remote CSVs with HTTP byte ranges and normalize matching rows",
    )
    sample_remote_news.add_argument("--url", required=True, help="Direct HTTP(S) CSV URL")
    sample_remote_news.add_argument(
        "--output",
        default=Path("data/raw/news.csv"),
        type=Path,
        help="Output normalized news CSV",
    )
    sample_remote_news.add_argument("--ticker-column", required=True, help="Input column for ticker")
    sample_remote_news.add_argument("--date-column", required=True, help="Input column for publish date")
    sample_remote_news.add_argument("--title-column", required=True, help="Input column for title/headline")
    sample_remote_news.add_argument("--summary-column", default=None, help="Optional summary/body column")
    sample_remote_news.add_argument("--source-column", default=None, help="Optional source/provider column")
    sample_remote_news.add_argument("--url-column", default=None, help="Optional article URL column")
    sample_remote_news.add_argument(
        "--source-name",
        default="remote-sample",
        help="Source value when no source column is provided",
    )
    sample_remote_news.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker allowlist, e.g. AAPL,MSFT,NVDA",
    )
    sample_remote_news.add_argument(
        "--range-bytes",
        default=25_000_000,
        type=int,
        help="Bytes to request per sampled HTTP range",
    )
    sample_remote_news.add_argument(
        "--step-bytes",
        default=None,
        type=int,
        help="Byte distance between sampled ranges; defaults to --range-bytes",
    )
    sample_remote_news.add_argument(
        "--max-ranges",
        default=20,
        type=int,
        help="Maximum number of HTTP ranges to sample",
    )
    sample_remote_news.add_argument(
        "--max-rows",
        default=5_000,
        type=int,
        help="Maximum normalized rows to keep",
    )
    sample_remote_news.add_argument(
        "--start-byte",
        default=0,
        type=int,
        help="First byte offset to sample from",
    )
    sample_remote_news.add_argument(
        "--timeout-seconds",
        default=120,
        type=int,
        help="Timeout per HTTP range request",
    )
    sample_remote_news.add_argument(
        "--min-rows-per-ticker",
        default=None,
        type=int,
        help="Keep scanning ranges until each requested ticker has at least this many rows",
    )
    sample_remote_news.set_defaults(func=sample_remote_news_command)

    build_dataset = subparsers.add_parser(
        "build-dataset", help="Join news and prices into ML rows"
    )
    build_dataset.add_argument("--news", required=True, type=Path, help="Input news CSV")
    build_dataset.add_argument(
        "--prices", required=True, type=Path, help="Input normalized prices CSV"
    )
    build_dataset.add_argument(
        "--output",
        default=Path("data/processed/modeling_dataset.csv"),
        type=Path,
        help="Output dataset CSV",
    )
    build_dataset.add_argument("--horizon-days", default=1, type=int)
    build_dataset.add_argument("--neutral-threshold", default=0.0025, type=float)
    build_dataset.set_defaults(func=build_dataset_command)

    train_baseline = subparsers.add_parser(
        "train-baseline",
        help="Train and evaluate a TF-IDF + logistic regression baseline model",
    )
    train_baseline.add_argument(
        "--dataset",
        default=Path("data/processed/modeling_dataset.csv"),
        type=Path,
        help="Input processed modeling dataset CSV",
    )
    train_baseline.add_argument(
        "--metrics-output",
        default=Path("reports/baseline_metrics.json"),
        type=Path,
        help="Output JSON metrics path",
    )
    train_baseline.add_argument(
        "--predictions-output",
        default=Path("reports/baseline_predictions.csv"),
        type=Path,
        help="Optional output CSV path for held-out predictions",
    )
    train_baseline.add_argument(
        "--model-output",
        default=None,
        type=Path,
        help="Optional joblib path for the trained sklearn pipeline",
    )
    train_baseline.add_argument("--test-size", default=0.25, type=float)
    train_baseline.add_argument("--max-features", default=5000, type=int)
    train_baseline.set_defaults(func=train_baseline_command)

    backtest_predictions = subparsers.add_parser(
        "backtest-predictions",
        help="Backtest held-out model predictions as simple long/cash strategies",
    )
    backtest_predictions.add_argument(
        "--predictions",
        default=Path("reports/baseline_predictions.csv"),
        type=Path,
        help="Input predictions CSV produced by train-baseline",
    )
    backtest_predictions.add_argument(
        "--summary-output",
        default=Path("reports/backtest_summary.csv"),
        type=Path,
        help="Output strategy summary CSV",
    )
    backtest_predictions.add_argument(
        "--trades-output",
        default=Path("reports/backtest_trades.csv"),
        type=Path,
        help="Optional output trades CSV",
    )
    backtest_predictions.add_argument(
        "--equity-output",
        default=Path("reports/backtest_equity_curve.csv"),
        type=Path,
        help="Optional output equity curve CSV",
    )
    backtest_predictions.add_argument("--initial-cash", default=100_000.0, type=float)
    backtest_predictions.add_argument("--transaction-cost", default=0.0, type=float)
    backtest_predictions.add_argument(
        "--slippage-pct",
        default=0.0,
        type=float,
        help="Per-trade slippage percentage as a decimal, e.g. 0.001 for 0.1%%",
    )
    backtest_predictions.set_defaults(func=backtest_predictions_command)

    init_db = subparsers.add_parser(
        "init-db",
        help="Create or update the local SQLite app database from database/schema.sql",
    )
    init_db.add_argument(
        "--database",
        default=DEFAULT_DATABASE_PATH,
        type=Path,
        help="SQLite database path to create/update",
    )
    init_db.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA_PATH,
        type=Path,
        help="Schema SQL file to execute",
    )
    init_db.set_defaults(func=init_db_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
