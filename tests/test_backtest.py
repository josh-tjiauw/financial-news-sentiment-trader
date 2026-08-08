from datetime import date

import pandas as pd
import pytest

from trading_sentiment.backtest import (
    Portfolio,
    apply_signal,
    backtest_predictions,
    calculate_max_drawdown,
    calculate_completed_trade_metrics,
    calculate_sharpe_like_return,
    prediction_label_to_signal,
)
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


def test_prediction_label_to_signal():
    assert prediction_label_to_signal(1) is Signal.BUY
    assert prediction_label_to_signal(0) is Signal.HOLD
    assert prediction_label_to_signal(-1) is Signal.SELL


def test_backtest_predictions_summarizes_strategy_against_buy_hold():
    predictions = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2024-01-02", "close": 100, "predicted_label": 1},
            {"ticker": "AAPL", "date": "2024-01-03", "close": 110, "predicted_label": -1},
            {"ticker": "MSFT", "date": "2024-01-02", "close": 200, "predicted_label": 0},
            {"ticker": "MSFT", "date": "2024-01-03", "close": 220, "predicted_label": 1},
        ]
    )

    summary, trades, equity_curve = backtest_predictions(predictions, initial_cash=1000)

    apple = summary.loc[summary["ticker"] == "AAPL"].iloc[0]
    assert apple["final_equity"] == 1100
    assert apple["strategy_return"] == 0.1
    assert apple["trade_count"] == 2
    assert apple["completed_trade_count"] == 1
    assert apple["winning_trade_count"] == 1
    assert apple["completed_trade_win_rate"] == 1.0
    assert apple["avg_completed_trade_return"] == 0.1
    assert apple["daily_win_rate"] == 1.0
    assert apple["exposure_pct"] == 0.5
    assert len(trades) == 3
    assert trades["completed_trade_pnl"].dropna().tolist() == [100.0]
    assert len(equity_curve) == 4
    assert set(equity_curve.columns) >= {"equity", "daily_return", "buy_hold_equity"}


def test_backtest_predictions_reports_completed_trade_win_loss_metrics():
    predictions = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2024-01-02", "close": 100, "predicted_label": 1},
            {"ticker": "AAPL", "date": "2024-01-03", "close": 110, "predicted_label": -1},
            {"ticker": "AAPL", "date": "2024-01-04", "close": 120, "predicted_label": 1},
            {"ticker": "AAPL", "date": "2024-01-05", "close": 108, "predicted_label": -1},
        ]
    )

    summary, trades, _ = backtest_predictions(predictions, initial_cash=1200)

    apple = summary.iloc[0]
    assert apple["completed_trade_count"] == 2
    assert apple["winning_trade_count"] == 1
    assert apple["losing_trade_count"] == 1
    assert apple["completed_trade_win_rate"] == 0.5
    assert apple["best_completed_trade_return"] == 0.1
    assert apple["worst_completed_trade_return"] == pytest.approx(-0.1)
    assert apple["total_completed_trade_pnl"] == pytest.approx(-12)
    assert trades["completed_trade_return"].dropna().tolist() == pytest.approx([0.1, -0.1])


def test_calculate_completed_trade_metrics_includes_transaction_costs():
    portfolio = Portfolio(cash=1000)
    apply_signal(portfolio, Signal.BUY, "QQQ", 100, date(2024, 1, 1), transaction_cost=5)
    apply_signal(portfolio, Signal.SELL, "QQQ", 110, date(2024, 1, 2), transaction_cost=5)

    metrics = calculate_completed_trade_metrics(portfolio.trades, transaction_cost=5)

    assert metrics["completed_trade_count"] == 1
    assert metrics["winning_trade_count"] == 1
    assert metrics["total_completed_trade_pnl"] == pytest.approx(89.5)


def test_backtest_predictions_requires_close_prices():
    predictions = pd.DataFrame(
        [{"ticker": "AAPL", "date": "2024-01-02", "predicted_label": 1}]
    )

    with pytest.raises(ValueError, match="close"):
        backtest_predictions(predictions)


def test_calculate_max_drawdown():
    equity = pd.Series([1000, 1200, 900, 1100])

    assert calculate_max_drawdown(equity) == -0.25


def test_calculate_sharpe_like_return_handles_flat_returns():
    returns = pd.Series([0.0, 0.0, 0.0])

    assert calculate_sharpe_like_return(returns) == 0.0
