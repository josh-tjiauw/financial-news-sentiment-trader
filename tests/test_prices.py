from pathlib import Path

from trading_sentiment.prices import load_prices_csv


def test_load_prices_csv_normalizes_ticker_and_date(tmp_path: Path):
    prices_csv = tmp_path / "prices.csv"
    prices_csv.write_text(
        "ticker,date,open,high,low,close,adj_close,volume\n"
        "aapl,2024-01-02,100,101,99,100,100,1000\n",
        encoding="utf-8",
    )

    prices = load_prices_csv(prices_csv)

    assert prices.loc[0, "ticker"] == "AAPL"
    assert str(prices.loc[0, "date"]) == "2024-01-02"
