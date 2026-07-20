"""
utils/database.py
=================
SQLite database setup and customer prediction logging.

Tables:
    customers — stores every new customer entry + their predicted segment.
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "models", "customers.db")


# ─── Cluster name mapping ─────────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Careful Spenders",
    1: "Budget Shoppers",
    2: "Impulsive Buyers",
    3: "Target Customers",
    4: "Average Customers",
}


# ─── 1. Initialise Database ───────────────────────────────────────────────────
def init_db() -> None:
    """
    Create the database file and the 'customers' table if they don't exist.

    Schema:
        id            INTEGER  — auto-incrementing primary key
        name          TEXT     — customer name (optional)
        gender        TEXT     — 'Male' or 'Female'
        age           INTEGER  — customer age
        annual_income REAL     — annual income in thousands
        spending_score INTEGER — spending score (1–100)
        cluster_id    INTEGER  — predicted cluster (0–4)
        cluster_name  TEXT     — human-readable cluster label
        added_at      TEXT     — ISO timestamp of record creation
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    DEFAULT 'Anonymous',
            gender         TEXT,
            age            INTEGER,
            annual_income  REAL,
            spending_score INTEGER,
            cluster_id     INTEGER,
            cluster_name   TEXT,
            added_at       TEXT
        )
    """)
    conn.commit()
    conn.close()


# ─── 2. Save Customer ─────────────────────────────────────────────────────────
def save_customer(name: str, gender: str, age: int,
                  annual_income: float, spending_score: int,
                  cluster_id: int) -> int:
    """
    Insert a new customer record and return the new row's ID.

    Args:
        name, gender, age, annual_income, spending_score — customer attributes
        cluster_id — predicted cluster from K-Means model

    Returns:
        int — the new record's ID
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO customers
            (name, gender, age, annual_income, spending_score, cluster_id, cluster_name, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        gender,
        int(age),
        float(annual_income),
        int(spending_score),
        int(cluster_id),
        CLUSTER_NAMES.get(cluster_id, "Unknown"),
        datetime.utcnow().isoformat(timespec="seconds"),
    ))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


# ─── 3. Retrieve All Customers ────────────────────────────────────────────────
def get_all_customers() -> list[dict]:
    """
    Fetch all rows from the customers table ordered by most recent first.

    Returns:
        list of dicts — one dict per customer row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row        # Access columns by name
    cur  = conn.cursor()
    cur.execute("SELECT * FROM customers ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ─── 4. Delete a Customer ────────────────────────────────────────────────────
def delete_customer(customer_id: int) -> None:
    """Delete the row with the given ID from the customers table."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DELETE FROM customers WHERE id = ?", (int(customer_id),))
    conn.commit()
    conn.close()


# ─── 5. Get Customer Count per Cluster ───────────────────────────────────────
def get_cluster_counts() -> dict:
    """Return {cluster_id: count} for all rows in the DB."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT cluster_id, COUNT(*) as cnt FROM customers GROUP BY cluster_id")
    result = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return result


# ─── 6. Get Total Count ──────────────────────────────────────────────────────
def get_total_count() -> int:
    """Return the total number of customers in the DB."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM customers")
    count = cur.fetchone()[0]
    conn.close()
    return int(count)
