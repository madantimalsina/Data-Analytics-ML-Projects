"""Small helper for running .sql files against CSV data using an in-memory SQLite db.

Each CSV is loaded into a table named after the file (without extension), so
`data/CA_housing.csv` becomes table `CA_housing`. SQL files in sql/ reference
these table names directly.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def load_csvs_to_sqlite(csv_paths: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for csv_path in csv_paths:
        table_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, index=False, if_exists="replace")
    return conn


def run_sql_file(conn: sqlite3.Connection, sql_path: str) -> pd.DataFrame:
    query = Path(sql_path).read_text()
    return pd.read_sql_query(query, conn)


def query_csv(sql_path: str, csv_paths: list[str]) -> pd.DataFrame:
    """Load csv_paths into SQLite tables and run the query in sql_path."""
    conn = load_csvs_to_sqlite(csv_paths)
    try:
        return run_sql_file(conn, sql_path)
    finally:
        conn.close()
