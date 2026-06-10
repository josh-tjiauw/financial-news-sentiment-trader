from pathlib import Path

import pandas as pd

_PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV price history from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install project dependencies with `pip install -e .[dev]` first") from exc

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker} from {start} to {end}")

    # yfinance may return a MultiIndex when multiple tickers are requested elsewhere.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    prices["ticker"] = ticker.upper()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices[_PRICE_COLUMNS]


def fetch_many_price_histories(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch and combine price history for multiple tickers."""
    frames = [fetch_price_history(ticker, start, end) for ticker in tickers]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_PRICE_COLUMNS)


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load normalized OHLCV prices from CSV."""
    prices = pd.read_csv(path)
    missing = set(_PRICE_COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"Price CSV missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices[_PRICE_COLUMNS]


def save_prices_csv(prices: pd.DataFrame, path: str | Path) -> None:
    """Persist normalized prices to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
