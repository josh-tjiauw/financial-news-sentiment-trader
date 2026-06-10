import argparse
from pathlib import Path

from trading_sentiment.dataset import build_dataset_from_csv
from trading_sentiment.prices import fetch_many_price_histories, save_prices_csv


def _split_tickers(value: str) -> list[str]:
    return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]


def fetch_prices_command(args: argparse.Namespace) -> None:
    prices = fetch_many_price_histories(_split_tickers(args.tickers), args.start, args.end)
    save_prices_csv(prices, args.output)
    print(f"Saved {len(prices)} price rows to {args.output}")


def build_dataset_command(args: argparse.Namespace) -> None:
    dataset = build_dataset_from_csv(
        news_csv=args.news,
        prices_csv=args.prices,
        output_csv=args.output,
        horizon_days=args.horizon_days,
        neutral_threshold=args.neutral_threshold,
    )
    print(f"Saved {len(dataset)} modeling rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-sentiment",
        description="Financial news sentiment trading research toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_prices = subparsers.add_parser("fetch-prices", help="Download OHLCV prices")
    fetch_prices.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    fetch_prices.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    fetch_prices.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    fetch_prices.add_argument(
        "--output",
        default=Path("data/raw/prices.csv"),
        type=Path,
        help="Output CSV path",
    )
    fetch_prices.set_defaults(func=fetch_prices_command)

    build_dataset = subparsers.add_parser("build-dataset", help="Join news and prices into ML rows")
    build_dataset.add_argument("--news", required=True, type=Path, help="Input news CSV")
    build_dataset.add_argument("--prices", required=True, type=Path, help="Input normalized prices CSV")
    build_dataset.add_argument(
        "--output",
        default=Path("data/processed/modeling_dataset.csv"),
        type=Path,
        help="Output dataset CSV",
    )
    build_dataset.add_argument("--horizon-days", default=1, type=int)
    build_dataset.add_argument("--neutral-threshold", default=0.0025, type=float)
    build_dataset.set_defaults(func=build_dataset_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
