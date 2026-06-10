from collections.abc import Sequence

from trading_sentiment.schemas import MovementLabel, PriceBar, Prediction, Signal


def future_return_label(
    current_close: float,
    future_close: float,
    neutral_threshold: float = 0.0025,
) -> MovementLabel:
    """Label future price movement using a neutral band to ignore tiny moves."""
    if current_close <= 0:
        raise ValueError("current_close must be greater than zero")

    pct_change = (future_close - current_close) / current_close
    if pct_change > neutral_threshold:
        return MovementLabel.POSITIVE
    if pct_change < -neutral_threshold:
        return MovementLabel.NEGATIVE
    return MovementLabel.NEUTRAL


def generate_price_labels(
    prices: Sequence[PriceBar], horizon_days: int = 1, neutral_threshold: float = 0.0025
) -> list[Prediction]:
    """Create labels from future close prices for each ticker independently."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    by_ticker: dict[str, list[PriceBar]] = {}
    for bar in prices:
        by_ticker.setdefault(bar.ticker, []).append(bar)

    labels: list[Prediction] = []
    for ticker, bars in by_ticker.items():
        ordered = sorted(bars, key=lambda bar: bar.date)
        for idx, bar in enumerate(ordered[:-horizon_days]):
            future = ordered[idx + horizon_days]
            label = future_return_label(bar.close, future.close, neutral_threshold)
            labels.append(Prediction(ticker=ticker, date=bar.date, score=float(label), label=label))

    return labels


def signal_from_score(score: float, buy_threshold: float = 0.35, sell_threshold: float = -0.35) -> Signal:
    """Convert a model score into a trading signal."""
    if score >= buy_threshold:
        return Signal.BUY
    if score <= sell_threshold:
        return Signal.SELL
    return Signal.HOLD
