import os
import sqlite3
import psycopg

DB_URI = os.getenv("DATABASE_URL")
PLACEHOLDER = "%s" if DB_URI else "?"


def _get_connection():
    if DB_URI:
        return psycopg.connect(DB_URI)
    return sqlite3.connect("usage.db")


def init_usage_table():
    conn = _get_connection()
    try:
        if DB_URI:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id SERIAL PRIMARY KEY,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd NUMERIC(12, 6) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def record_usage(input_tokens: int, output_tokens: int, cost_usd: float):
    conn = _get_connection()
    try:
        conn.execute(
            f"""
            INSERT INTO token_usage (input_tokens, output_tokens, cost_usd)
            VALUES ({PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER})
            """,
            (input_tokens, output_tokens, cost_usd),
        )
        conn.commit()
    finally:
        conn.close()


def get_usage_totals() -> dict:
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(cost_usd), 0)
            FROM token_usage
            """
        ).fetchone()
    finally:
        conn.close()

    input_tokens, output_tokens, cost_usd = row
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens),
        "estimated_cost_usd": round(float(cost_usd), 6),
    }
