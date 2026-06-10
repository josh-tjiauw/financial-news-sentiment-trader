from datetime import date

import pytest

from trading_sentiment.labeling import future_return_label, generate_price_labels, signal_from_score
from trading_sentiment.schemas import MovementLabel, PriceBar, Signal


def test_future_return_label_positive_negative_neutral():
    assert future_return_label(100, 102) is MovementLabel.POSITIVE
    assert future_return_label(100, 98) is MovementLabel.NEGATIVE
    assert future_return_label(100, 100.1) is MovementLabel.NEUTRAL


def test_future_return_label_rejects_bad_price():
    with pytest.raises(ValueError):
        future_return_label(0, 100)


def test_generate_price_labels_uses_future_price():
    prices = [
        PriceBar("AAPL", date(2024, 1, 1), 100, 101, 99, 100),
        PriceBar("AAPL", date(2024, 1, 2), 101, 103, 100, 103),
    ]
    labels = generate_price_labels(prices)
    assert labels[0].label is MovementLabel.POSITIVE


def test_signal_from_score():
    assert signal_from_score(0.8) is Signal.BUY
    assert signal_from_score(-0.8) is Signal.SELL
    assert signal_from_score(0.0) is Signal.HOLD
