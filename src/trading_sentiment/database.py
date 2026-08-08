from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "app" / "trading_sentiment.sqlite"


def connect_database(database_path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite app database with project defaults enabled."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> Path:
    """Create or migrate the local SQLite database using the project schema."""
    database_path = Path(database_path)
    schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file does not exist: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")
    with connect_database(database_path) as connection:
        connection.executescript(schema_sql)

    return database_path


def list_tables(database_path: Path = DEFAULT_DATABASE_PATH) -> list[str]:
    """Return user-created table names in the local app database."""
    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    return [row["name"] for row in rows]
