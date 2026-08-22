import os
import psycopg


def _get_connection():
    return psycopg.connect(os.environ["DATABASE_URL"])


def init_usage_table():
    with _get_connection() as conn:
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


def record_usage(input_tokens: int, output_tokens: int, cost_usd: float):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO token_usage (input_tokens, output_tokens, cost_usd)
            VALUES (%s, %s, %s)
            """,
            (input_tokens, output_tokens, cost_usd),
        )


def get_usage_totals() -> dict:
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(cost_usd), 0)
            FROM token_usage
            """
        ).fetchone()

    input_tokens, output_tokens, cost_usd = row
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens),
        "estimated_cost_usd": round(float(cost_usd), 6),
    }
