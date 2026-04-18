"""SQLite storage utilities for experiment runs."""

from __future__ import annotations

import sqlite3
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = Path("results") / "experiment_results.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_parent_dir(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            mode TEXT NOT NULL,
            model TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS experiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            detected_drift_type TEXT,
            role TEXT NOT NULL,
            condition TEXT NOT NULL,
            expected_drift INTEGER NOT NULL,
            drift_detected INTEGER NOT NULL,
            blocked INTEGER NOT NULL,
            ids_score REAL,
            base_ids REAL,
            mahalanobis REAL,
            kl_divergence REAL,
            js_divergence REAL,
            wasserstein REAL,
            hellinger REAL,
            tool_frequency REAL,
            decision TEXT,
            score REAL NOT NULL,
            latency_ms INTEGER NOT NULL,
            response_excerpt TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_results_run_id
            ON experiment_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_results_prompt_type
            ON experiment_results(prompt_type);
        CREATE INDEX IF NOT EXISTS idx_results_detected_drift_type
            ON experiment_results(detected_drift_type);
        CREATE INDEX IF NOT EXISTS idx_results_role_condition
            ON experiment_results(role, condition);
        """
    )
    _migrate_experiment_results(conn)
    _migrate_experiment_runs(conn)
    conn.commit()


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(r[1]) == column for r in rows)


def _migrate_experiment_results(conn: sqlite3.Connection) -> None:
    """Add IDS component columns on existing databases (idempotent)."""
    for col, sql_type in (
        ("ids_score", "REAL"),
        ("base_ids", "REAL"),
        ("mahalanobis", "REAL"),
        ("kl_divergence", "REAL"),
        ("js_divergence", "REAL"),
        ("wasserstein", "REAL"),
        ("hellinger", "REAL"),
        ("tool_frequency", "REAL"),
    ):
        if not _table_has_column(conn, "experiment_results", col):
            conn.execute(
                f"ALTER TABLE experiment_results ADD COLUMN {col} {sql_type}"
            )


def _migrate_experiment_runs(conn: sqlite3.Connection) -> None:
    if not _table_has_column(conn, "experiment_runs", "model"):
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN model TEXT")


def create_run(conn: sqlite3.Connection, run_id: str, mode: str, model: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO experiment_runs (run_id, mode, model, started_at)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, mode, model, utc_now_iso()),
    )
    conn.commit()


def complete_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        """
        UPDATE experiment_runs
        SET completed_at = ?
        WHERE run_id = ?
        """,
        (utc_now_iso(), run_id),
    )
    conn.commit()


def insert_result(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO experiment_results (
            run_id, prompt_id, prompt_type, detected_drift_type, role, condition,
            expected_drift, drift_detected, blocked, score, latency_ms,
            ids_score, base_ids, mahalanobis, kl_divergence, js_divergence,
            wasserstein, hellinger, tool_frequency, decision, response_excerpt, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["run_id"],
            row["prompt_id"],
            row["prompt_type"],
            row.get("detected_drift_type"),
            row["role"],
            row["condition"],
            row["expected_drift"],
            row["drift_detected"],
            row["blocked"],
            row["score"],
            row["latency_ms"],
            row.get("base_ids", row.get("ids_score")),
            row.get("base_ids"),
            row.get("mahalanobis"),
            row.get("kl_divergence"),
            row.get("js_divergence"),
            row.get("wasserstein"),
            row.get("hellinger"),
            row.get("tool_frequency"),
            row.get("decision"),
            row.get("response_excerpt", ""),
            row["created_at"],
        ),
    )


def insert_results(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        insert_result(conn, row)
    conn.commit()


def _result_columns(conn: sqlite3.Connection) -> list[str]:
    info = conn.execute("PRAGMA table_info(experiment_results)").fetchall()
    return [str(row["name"]) for row in info]


def export_run_to_csv(conn: sqlite3.Connection, run_id: str, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"
    cols = _result_columns(conn)
    rows = conn.execute(
        "SELECT * FROM experiment_results WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([row[col] for col in cols])
    return out_path

