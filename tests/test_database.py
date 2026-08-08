from __future__ import annotations

import sqlite3
from pathlib import Path

from trading_sentiment.database import connect_database, initialize_database, list_tables


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "database" / "schema.sql"


def test_initialize_database_creates_schema_and_seed(tmp_path: Path) -> None:
    database_path = tmp_path / "app" / "trading_sentiment.sqlite"

    initialized_path = initialize_database(database_path=database_path, schema_path=SCHEMA_PATH)

    assert initialized_path == database_path
    assert database_path.exists()

    with connect_database(database_path) as connection:
        row = connection.execute(
            "SELECT symbol, name FROM securities WHERE symbol = ?",
            ("FXAIX",),
        ).fetchone()

    assert dict(row) == {"symbol": "FXAIX", "name": "Fidelity 500 Index Fund"}


def test_initialize_database_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "trading_sentiment.sqlite"

    initialize_database(database_path=database_path, schema_path=SCHEMA_PATH)
    initialize_database(database_path=database_path, schema_path=SCHEMA_PATH)

    with connect_database(database_path) as connection:
        security_count = connection.execute("SELECT COUNT(*) FROM securities").fetchone()[0]

    assert security_count == 1


def test_connect_database_enables_foreign_keys(tmp_path: Path) -> None:
    database_path = tmp_path / "trading_sentiment.sqlite"
    initialize_database(database_path=database_path, schema_path=SCHEMA_PATH)

    with connect_database(database_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO price_bars (security_id, price_date, close_price)
                VALUES (999999, '2026-06-15', 100)
                """
            )
        except sqlite3.IntegrityError:
            pass
        else:  # pragma: no cover - fail clearly if foreign keys are disabled
            raise AssertionError("foreign key enforcement should be enabled")


def test_list_tables_returns_app_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "trading_sentiment.sqlite"
    initialize_database(database_path=database_path, schema_path=SCHEMA_PATH)

    tables = list_tables(database_path)

    assert "securities" in tables
    assert "trades" in tables
    assert "portfolio_snapshots" in tables
