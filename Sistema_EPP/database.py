from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "epp_events.db"


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS workers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE,
                full_name TEXT NOT NULL,
                area TEXT,
                face_template TEXT,
                face_image_path TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                camera_name TEXT NOT NULL,
                worker_id INTEGER,
                worker_label TEXT NOT NULL DEFAULT 'No identificado',
                violation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(camera_id) REFERENCES cameras(id),
                FOREIGN KEY(worker_id) REFERENCES workers(id)
            );

            CREATE TABLE IF NOT EXISTS access_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                camera_name TEXT NOT NULL,
                person_type TEXT NOT NULL,
                worker_id INTEGER,
                person_label TEXT NOT NULL,
                helmet_ok INTEGER NOT NULL,
                vest_ok INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(camera_id) REFERENCES cameras(id),
                FOREIGN KEY(worker_id) REFERENCES workers(id)
            );
            """
        )

        if "is_access_control" not in [row["name"] for row in conn.execute("PRAGMA table_info(cameras)")]:
            conn.execute("ALTER TABLE cameras ADD COLUMN is_access_control INTEGER DEFAULT 0")

        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workers)")]
        if "face_template" not in columns:
            conn.execute("ALTER TABLE workers ADD COLUMN face_template TEXT")
        if "face_image_path" not in columns:
            conn.execute("ALTER TABLE workers ADD COLUMN face_image_path TEXT")
        if "person_type" not in columns:
            conn.execute("ALTER TABLE workers ADD COLUMN person_type TEXT DEFAULT 'Personal'")


def list_cameras(active_only: bool = True) -> list[sqlite3.Row]:
    query = "SELECT * FROM cameras"
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY id"
    with connect() as conn:
        return list(conn.execute(query))


def add_camera(name: str, source: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO cameras (name, source) VALUES (?, ?)", (name, source))


def delete_camera(camera_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE cameras SET active = 0 WHERE id = ?", (camera_id,))


def add_worker(code: str, full_name: str, area: str, person_type: str = "Personal") -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO workers (code, full_name, area, person_type)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                full_name = excluded.full_name,
                area = excluded.area,
                person_type = excluded.person_type,
                active = 1
            """,
            (code or None, full_name, area, person_type),
        )


def delete_worker(worker_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE workers SET active = 0 WHERE id = ?", (worker_id,))


def list_workers() -> list[sqlite3.Row]:
    with connect() as conn:
        return list(conn.execute("SELECT * FROM workers WHERE active = 1 ORDER BY full_name"))


def get_worker(worker_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM workers WHERE id = ? AND active = 1", (worker_id,)).fetchone()


def save_worker_face(worker_id: int, face_template: str, face_image_path: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE workers
            SET face_template = ?, face_image_path = ?
            WHERE id = ?
            """,
            (face_template, face_image_path, worker_id),
        )


def log_violation(
    camera_id: int | None,
    camera_name: str,
    violation_type: str,
    confidence: float,
    snapshot_path: str | None,
    worker_id: int | None = None,
    worker_label: str = "No identificado",
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO violations (
                camera_id, camera_name, worker_id, worker_label,
                violation_type, confidence, snapshot_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                camera_name,
                worker_id,
                worker_label,
                violation_type,
                confidence,
                snapshot_path,
            ),
        )


def list_violations(limit: int = 100) -> list[sqlite3.Row]:
    with connect() as conn:
        return list(
            conn.execute(
                """
                SELECT *
                FROM violations
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        )


def count_by_type() -> Iterable[sqlite3.Row]:
    with connect() as conn:
        return list(
            conn.execute(
                """
                SELECT violation_type, COUNT(*) AS total
                FROM violations
                GROUP BY violation_type
                ORDER BY total DESC
                """
            )
        )


def log_access_event(
    camera_id: int | None,
    camera_name: str,
    person_type: str,
    person_label: str,
    helmet_ok: bool,
    vest_ok: bool,
    decision: str,
    reason: str,
    worker_id: int | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO access_events (
                camera_id, camera_name, person_type, worker_id, person_label,
                helmet_ok, vest_ok, decision, reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                camera_name,
                person_type,
                worker_id,
                person_label,
                int(helmet_ok),
                int(vest_ok),
                decision,
                reason,
            ),
        )


def list_access_events(limit: int = 100) -> list[sqlite3.Row]:
    with connect() as conn:
        return list(
            conn.execute(
                """
                SELECT *
                FROM access_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        )
