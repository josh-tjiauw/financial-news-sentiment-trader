from datetime import date

import pandas as pd

from trading_sentiment.dataset import build_dataset


def test_build_dataset_joins_news_prices_and_labels_future_return():
    news = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_date": date(2024, 1, 2),
                "title": "Apple rises on AI demand",
                "summary": "Analysts expect growth",
                "source": "sample",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": date(2024, 1, 2),
                "open": 100,
                "high": 103,
                "low": 99,
                "close": 100,
                "adj_close": 100,
                "volume": 1000,
            },
            {
                "ticker": "AAPL",
                "date": date(2024, 1, 3),
                "open": 101,
                "high": 106,
                "low": 100,
                "close": 105,
                "adj_close": 105,
                "volume": 1100,
            },
        ]
    )

    dataset = build_dataset(news, prices)

    assert len(dataset) == 1
    assert dataset.loc[0, "future_close"] == 105
    assert dataset.loc[0, "future_return"] == 0.05
    assert dataset.loc[0, "label"] == 1
    assert dataset.loc[0, "cleaned_text"] == "apple rises ai demand analysts expect growth"
