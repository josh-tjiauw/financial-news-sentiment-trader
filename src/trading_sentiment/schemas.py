from dataclasses import dataclass
from datetime import date
from enum import Enum


class MovementLabel(int, Enum):
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class NewsArticle:
    ticker: str
    published_date: date
    title: str
    summary: str = ""
    source: str | None = None


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass(frozen=True)
class Prediction:
    ticker: str
    date: date
    score: float
    label: MovementLabel


@dataclass(frozen=True)
class Trade:
    date: date
    ticker: str
    signal: Signal
    price: float
    shares: float
    cash_after: float
    equity_after: float
