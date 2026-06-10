from dataclasses import dataclass, field
from datetime import date

from trading_sentiment.schemas import Signal, Trade


@dataclass
class Portfolio:
    cash: float = 100_000.0
    shares: float = 0.0
    ticker: str | None = None
    trades: list[Trade] = field(default_factory=list)

    def equity(self, current_price: float | None = None) -> float:
        if self.shares and current_price is None:
            raise ValueError("current_price is required when holding shares")
        return self.cash + (self.shares * (current_price or 0.0))


def apply_signal(
    portfolio: Portfolio,
    signal: Signal,
    ticker: str,
    price: float,
    trade_date: date,
    transaction_cost: float = 0.0,
) -> Portfolio:
    """Apply a simple long/cash signal to a portfolio.

    BUY invests all cash into the ticker. SELL liquidates the current position. HOLD does nothing.
    """
    if price <= 0:
        raise ValueError("price must be greater than zero")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")

    if signal is Signal.BUY and portfolio.cash > transaction_cost and portfolio.shares == 0:
        investable_cash = portfolio.cash - transaction_cost
        portfolio.shares = investable_cash / price
        portfolio.cash = 0.0
        portfolio.ticker = ticker
    elif signal is Signal.SELL and portfolio.shares > 0:
        portfolio.cash = (portfolio.shares * price) - transaction_cost
        portfolio.shares = 0.0
        portfolio.ticker = None
    else:
        return portfolio

    portfolio.trades.append(
        Trade(
            date=trade_date,
            ticker=ticker,
            signal=signal,
            price=price,
            shares=portfolio.shares,
            cash_after=portfolio.cash,
            equity_after=portfolio.equity(price if portfolio.shares else None),
        )
    )
    return portfolio
