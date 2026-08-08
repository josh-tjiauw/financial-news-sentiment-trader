from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

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
    slippage_pct: float = 0.0,
) -> Portfolio:
    """Apply a simple long/cash signal to a portfolio.

    BUY invests all cash into the ticker. SELL liquidates the current position. HOLD does nothing.
    """
    if price <= 0:
        raise ValueError("price must be greater than zero")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")
    if slippage_pct < 0:
        raise ValueError("slippage_pct cannot be negative")

    execution_price = price
    if signal is Signal.BUY:
        execution_price = price * (1 + slippage_pct)
    elif signal is Signal.SELL:
        execution_price = price * (1 - slippage_pct)

    if signal is Signal.BUY and portfolio.cash > transaction_cost and portfolio.shares == 0:
        investable_cash = portfolio.cash - transaction_cost
        portfolio.shares = investable_cash / execution_price
        portfolio.cash = 0.0
        portfolio.ticker = ticker
    elif signal is Signal.SELL and portfolio.shares > 0:
        portfolio.cash = (portfolio.shares * execution_price) - transaction_cost
        portfolio.shares = 0.0
        portfolio.ticker = None
    else:
        return portfolio

    portfolio.trades.append(
        Trade(
            date=trade_date,
            ticker=ticker,
            signal=signal,
            price=execution_price,
            shares=portfolio.shares,
            cash_after=portfolio.cash,
            equity_after=portfolio.equity(price if portfolio.shares else None),
        )
    )
    return portfolio


def prediction_label_to_signal(predicted_label: int) -> Signal:
    """Convert a movement prediction into a simple long/cash trading signal."""
    if predicted_label > 0:
        return Signal.BUY
    if predicted_label < 0:
        return Signal.SELL
    return Signal.HOLD


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate max drawdown from an equity series as a negative percentage."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


def calculate_sharpe_like_return(daily_returns: pd.Series) -> float:
    """Calculate a simple annualized Sharpe-like return/risk metric.

    This intentionally omits a risk-free rate so it stays dependency-free and easy to explain.
    """
    non_null_returns = daily_returns.dropna()
    if non_null_returns.empty or non_null_returns.std(ddof=0) == 0:
        return 0.0
    return float((non_null_returns.mean() / non_null_returns.std(ddof=0)) * np.sqrt(252))


def calculate_completed_trade_metrics(
    trades: list[Trade],
    transaction_cost: float = 0.0,
) -> dict[str, float | int]:
    """Summarize realized P/L for completed buy/sell trade pairs."""
    completed_returns: list[float] = []
    completed_pnls: list[float] = []
    open_buy: Trade | None = None

    for trade in trades:
        if trade.signal is Signal.BUY:
            open_buy = trade
        elif trade.signal is Signal.SELL and open_buy is not None:
            entry_value = open_buy.shares * open_buy.price
            gross_pnl = (trade.price - open_buy.price) * open_buy.shares
            net_pnl = gross_pnl - (transaction_cost * 2)
            completed_pnls.append(net_pnl)
            completed_returns.append(net_pnl / entry_value if entry_value else 0.0)
            open_buy = None

    winning_returns = [trade_return for trade_return in completed_returns if trade_return > 0]
    losing_returns = [trade_return for trade_return in completed_returns if trade_return < 0]
    completed_count = len(completed_returns)

    return {
        "completed_trade_count": completed_count,
        "winning_trade_count": len(winning_returns),
        "losing_trade_count": len(losing_returns),
        "completed_trade_win_rate": len(winning_returns) / completed_count
        if completed_count
        else 0.0,
        "avg_completed_trade_return": float(np.mean(completed_returns))
        if completed_returns
        else 0.0,
        "best_completed_trade_return": max(completed_returns) if completed_returns else 0.0,
        "worst_completed_trade_return": min(completed_returns) if completed_returns else 0.0,
        "total_completed_trade_pnl": float(sum(completed_pnls)),
    }


def backtest_predictions(
    predictions: pd.DataFrame,
    initial_cash: float = 100_000.0,
    transaction_cost: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest predicted labels as independent long/cash strategies per ticker.

    Required columns: ticker, date, close, predicted_label.
    Each ticker gets its own portfolio so the first version stays easy to inspect.
    """
    required_columns = {"ticker", "date", "close", "predicted_label"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"predictions are missing required columns: {missing}")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")
    if slippage_pct < 0:
        raise ValueError("slippage_pct cannot be negative")

    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.sort_values(["ticker", "date"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | int | str]] = []
    trade_rows: list[dict[str, float | str]] = []
    equity_rows: list[dict[str, float | str]] = []

    for ticker, ticker_rows in rows.groupby("ticker", sort=True):
        portfolio = Portfolio(cash=initial_cash)
        first_close = float(ticker_rows.iloc[0]["close"])
        last_close = float(ticker_rows.iloc[-1]["close"])
        previous_equity = initial_cash

        for _, row in ticker_rows.iterrows():
            signal = prediction_label_to_signal(int(row["predicted_label"]))
            close_price = float(row["close"])
            portfolio = apply_signal(
                portfolio=portfolio,
                signal=signal,
                ticker=str(ticker),
                price=close_price,
                trade_date=row["date"].date(),
                transaction_cost=transaction_cost,
                slippage_pct=slippage_pct,
            )
            equity = portfolio.equity(close_price)
            buy_hold_equity = initial_cash * (close_price / first_close)
            daily_return = (equity - previous_equity) / previous_equity
            equity_rows.append(
                {
                    "date": row["date"].date().isoformat(),
                    "ticker": str(ticker),
                    "close": close_price,
                    "predicted_label": int(row["predicted_label"]),
                    "signal": signal.value,
                    "cash": portfolio.cash,
                    "shares": portfolio.shares,
                    "equity": equity,
                    "daily_return": daily_return,
                    "buy_hold_equity": buy_hold_equity,
                    "buy_hold_return": (buy_hold_equity - initial_cash) / initial_cash,
                }
            )
            previous_equity = equity

        final_equity = portfolio.equity(last_close)
        buy_hold_final_equity = initial_cash * (last_close / first_close)
        ticker_equity = pd.DataFrame(
            [row for row in equity_rows if row["ticker"] == str(ticker)]
        )
        completed_trade_metrics = calculate_completed_trade_metrics(
            portfolio.trades,
            transaction_cost=transaction_cost,
        )
        positive_days = int((ticker_equity["daily_return"] > 0).sum())
        non_flat_days = int((ticker_equity["daily_return"] != 0).sum())

        summary_rows.append(
            {
                "ticker": str(ticker),
                "start_date": ticker_rows.iloc[0]["date"].date().isoformat(),
                "end_date": ticker_rows.iloc[-1]["date"].date().isoformat(),
                "initial_cash": initial_cash,
                "final_equity": final_equity,
                "strategy_return": (final_equity - initial_cash) / initial_cash,
                "buy_hold_final_equity": buy_hold_final_equity,
                "buy_hold_return": (buy_hold_final_equity - initial_cash) / initial_cash,
                "excess_return": (final_equity - buy_hold_final_equity) / initial_cash,
                "trade_count": len(portfolio.trades),
                "exposure_pct": float((ticker_equity["shares"] > 0).mean()),
                "positive_day_count": positive_days,
                "non_flat_day_count": non_flat_days,
                "daily_win_rate": positive_days / non_flat_days if non_flat_days else 0.0,
                "max_drawdown": calculate_max_drawdown(ticker_equity["equity"]),
                "buy_hold_max_drawdown": calculate_max_drawdown(ticker_equity["buy_hold_equity"]),
                "avg_daily_return": float(ticker_equity["daily_return"].mean()),
                "daily_volatility": float(ticker_equity["daily_return"].std(ddof=0)),
                "sharpe_like": calculate_sharpe_like_return(ticker_equity["daily_return"]),
                **completed_trade_metrics,
            }
        )

        open_buy: Trade | None = None
        for trade in portfolio.trades:
            completed_trade_pnl = None
            completed_trade_return = None
            if trade.signal is Signal.BUY:
                open_buy = trade
            elif trade.signal is Signal.SELL and open_buy is not None:
                entry_value = open_buy.shares * open_buy.price
                completed_trade_pnl = (
                    (trade.price - open_buy.price) * open_buy.shares
                ) - (transaction_cost * 2)
                completed_trade_return = (
                    completed_trade_pnl / entry_value if entry_value else 0.0
                )
                open_buy = None

            trade_rows.append(
                {
                    "date": trade.date.isoformat(),
                    "ticker": trade.ticker,
                    "signal": trade.signal.value,
                    "price": trade.price,
                    "shares_after": trade.shares,
                    "cash_after": trade.cash_after,
                    "equity_after": trade.equity_after,
                    "completed_trade_pnl": completed_trade_pnl,
                    "completed_trade_return": completed_trade_return,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(trade_rows), pd.DataFrame(equity_rows)


def backtest_predictions_from_csv(
    predictions_csv: str | Path,
    summary_output: str | Path,
    trades_output: str | Path | None = None,
    equity_output: str | Path | None = None,
    initial_cash: float = 100_000.0,
    transaction_cost: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prediction CSV artifacts, backtest them, and write report CSVs."""
    predictions = pd.read_csv(predictions_csv)
    summary, trades, equity_curve = backtest_predictions(
        predictions,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        slippage_pct=slippage_pct,
    )

    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    if trades_output is not None:
        trades_path = Path(trades_output)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(trades_path, index=False)

    if equity_output is not None:
        equity_path = Path(equity_output)
        equity_path.parent.mkdir(parents=True, exist_ok=True)
        equity_curve.to_csv(equity_path, index=False)

    return summary, trades, equity_curve
