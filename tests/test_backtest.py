from datetime import date

from trading_sentiment.backtest import Portfolio, apply_signal
from trading_sentiment.schemas import Signal


def test_apply_buy_signal_invests_cash():
    portfolio = apply_signal(Portfolio(cash=1000), Signal.BUY, "QQQ", 100, date(2024, 1, 1))
    assert portfolio.cash == 0
    assert portfolio.shares == 10
    assert portfolio.ticker == "QQQ"


def test_apply_sell_signal_liquidates_position():
    portfolio = Portfolio(cash=0, shares=10, ticker="QQQ")
    portfolio = apply_signal(portfolio, Signal.SELL, "QQQ", 110, date(2024, 1, 2))
    assert portfolio.cash == 1100
    assert portfolio.shares == 0
    assert portfolio.ticker is None
