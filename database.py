import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS contact_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(
    db_path: Path,
    filename: str,
    predicted_class: str,
    confidence: float,
    source: str,
) -> None:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (filename, predicted_class, confidence, source)
            VALUES (?, ?, ?, ?)
            """,
            (filename, predicted_class, confidence, source),
        )
        connection.commit()


def get_recent_predictions(db_path: Path, limit: int = 6) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT filename, predicted_class, confidence, source, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_all_predictions(db_path: Path, limit: int = 24) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT filename, predicted_class, confidence, source, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_prediction_summary(db_path: Path) -> dict:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT predicted_class, COUNT(*)
            FROM predictions
            GROUP BY predicted_class
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """
        )
        top_row = cursor.fetchone()
        return {
            "total_predictions": total_predictions,
            "top_class": top_row[0] if top_row else "No data yet",
            "top_class_count": top_row[1] if top_row else 0,
        }


def save_contact_message(db_path: Path, name: str, email: str, message: str) -> None:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO contact_messages (name, email, message)
            VALUES (?, ?, ?)
            """,
            (name, email, message),
        )
        connection.commit()
