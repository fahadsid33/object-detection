import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "incidents.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def initialize():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                class_name TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Needs review'
            )
        """)


def add_incidents(incidents):
    initialize()
    if not incidents:
        return
    with connect() as con:
        con.executemany("""
            INSERT INTO incidents
            (timestamp_utc, source, incident_type, severity, confidence, class_name, review_status)
            VALUES (:timestamp_utc, :source, :incident_type, :severity, :confidence, :class_name, :review_status)
        """, incidents)


def list_incidents(limit=500):
    initialize()
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]


def update_status(incident_id: int, status: str):
    allowed = {"Needs review", "Reviewed", "Dismissed"}
    if status not in allowed:
        raise ValueError("Invalid review status")
    initialize()
    with connect() as con:
        cur = con.execute(
            "UPDATE incidents SET review_status=? WHERE id=?",
            (status, int(incident_id))
        )
        return cur.rowcount > 0
