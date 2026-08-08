from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


def test_database_schema_executes_and_seeds_fxaix() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    connection.executescript(schema_sql)

    row = connection.execute(
        "SELECT symbol, name, asset_type, currency FROM securities WHERE symbol = ?",
        ("FXAIX",),
    ).fetchone()

    assert row is not None
    assert dict(row) == {
        "symbol": "FXAIX",
        "name": "Fidelity 500 Index Fund",
        "asset_type": "mutual_fund",
        "currency": "USD",
    }


def test_database_schema_enforces_trade_status_close_requirements() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    security_id = connection.execute(
        "SELECT id FROM securities WHERE symbol = ?",
        ("FXAIX",),
    ).fetchone()[0]

    connection.execute(
        """
        INSERT INTO trades (security_id, opened_at, direction, quantity, entry_price, status)
        VALUES (?, '2026-06-13T09:30:00', 'long', 1, 100, 'open')
        """,
        (security_id,),
    )

    try:
        connection.execute(
            """
            INSERT INTO trades (security_id, opened_at, direction, quantity, entry_price, status)
            VALUES (?, '2026-06-13T09:30:00', 'long', 1, 100, 'closed')
            """,
            (security_id,),
        )
    except sqlite3.IntegrityError:
        pass
    else:  # pragma: no cover - fail clearly if the constraint stops working
        raise AssertionError("closed trades should require closed_at and exit_price")
