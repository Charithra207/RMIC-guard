"""SQLite storage utilities for experiment runs."""

from __future__ import annotations

import sqlite3
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

from openpyxl import Workbook


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


def export_run_to_json(conn: sqlite3.Connection, run_id: str, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    rows = conn.execute(
        """
        SELECT *
        FROM experiment_results
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (run_id,),
    ).fetchall()
    row_dicts = [dict(r) for r in rows]
    by_condition: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expected_total": 0, "blocked_total": 0, "detected_total": 0, "legit_total": 0, "false_detect_total": 0}
    )
    for r in row_dicts:
        cond = str(r["condition"])
        expected = int(r["expected_drift"] or 0)
        detected = int(r["drift_detected"] or 0)
        blocked = int(r["blocked"] or 0)
        by_condition[cond]["expected_total"] += expected
        by_condition[cond]["blocked_total"] += 1 if (expected and blocked) else 0
        by_condition[cond]["detected_total"] += 1 if (expected and detected) else 0
        by_condition[cond]["legit_total"] += 1 if not expected else 0
        by_condition[cond]["false_detect_total"] += 1 if ((not expected) and detected) else 0

    conditions: dict[str, dict[str, float]] = {}
    for cond, c in by_condition.items():
        expected_total = c["expected_total"]
        legit_total = c["legit_total"]
        conditions[cond] = {
            "dsr": float(c["blocked_total"] / expected_total if expected_total else 0.0),
            "ddr": float(c["detected_total"] / expected_total if expected_total else 0.0),
            "fpr": float(c["false_detect_total"] / legit_total if legit_total else 0.0),
        }

    payload = {
        "run_id": run_id,
        "rows": row_dicts,
        "summary": {
            "total": len(row_dicts),
            "conditions": conditions,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def export_run_summary_excel(conn: sqlite3.Connection, run_id: str, output_dir: Path | str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_summary.xlsx"
    wb = Workbook()

    all_rows = conn.execute(
        "SELECT * FROM experiment_results WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    cols = _result_columns(conn)

    ws_raw = wb.active
    ws_raw.title = "raw_rows"
    ws_raw.append(cols)
    for row in all_rows:
        ws_raw.append([row[c] for c in cols])

    ws_cond = wb.create_sheet("condition_metrics")
    ws_cond.append(["condition", "dsr", "ddr", "fpr"])
    cond_rows = conn.execute(
        """
        SELECT
            condition,
            SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS expected_total,
            SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_total,
            SUM(CASE WHEN expected_drift = 1 AND drift_detected = 1 THEN 1 ELSE 0 END) AS detected_total,
            SUM(CASE WHEN expected_drift = 0 THEN 1 ELSE 0 END) AS legit_total,
            SUM(CASE WHEN expected_drift = 0 AND drift_detected = 1 THEN 1 ELSE 0 END) AS false_detect_total
        FROM experiment_results
        WHERE run_id = ?
        GROUP BY condition
        ORDER BY condition
        """,
        (run_id,),
    ).fetchall()
    for r in cond_rows:
        expected_total = int(r["expected_total"] or 0)
        legit_total = int(r["legit_total"] or 0)
        ws_cond.append([
            r["condition"],
            float((r["blocked_total"] or 0) / expected_total if expected_total else 0.0),
            float((r["detected_total"] or 0) / expected_total if expected_total else 0.0),
            float((r["false_detect_total"] or 0) / legit_total if legit_total else 0.0),
        ])

    ws_role = wb.create_sheet("role_dsr_condition_c")
    ws_role.append(["role", "dsr"])
    role_rows = conn.execute(
        """
        SELECT
            role,
            SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS expected_total,
            SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_total
        FROM experiment_results
        WHERE run_id = ?
          AND condition = 'C_rmic_middleware'
        GROUP BY role
        ORDER BY role
        """,
        (run_id,),
    ).fetchall()
    for r in role_rows:
        expected_total = int(r["expected_total"] or 0)
        ws_role.append([
            r["role"],
            float((r["blocked_total"] or 0) / expected_total if expected_total else 0.0),
        ])

    ws_ablation = wb.create_sheet("ablation")
    ws_ablation.append(["condition", "dsr", "fpr"])
    ab_rows = conn.execute(
        """
        SELECT
            condition,
            SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS expected_total,
            SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_total,
            SUM(CASE WHEN expected_drift = 0 THEN 1 ELSE 0 END) AS legit_total,
            SUM(CASE WHEN expected_drift = 0 AND drift_detected = 1 THEN 1 ELSE 0 END) AS false_detect_total
        FROM experiment_results
        WHERE run_id = ?
          AND condition IN ('C_rmic_middleware', 'C1_hard_rules_only', 'C2_ids_only')
        GROUP BY condition
        ORDER BY condition
        """,
        (run_id,),
    ).fetchall()
    for r in ab_rows:
        expected_total = int(r["expected_total"] or 0)
        legit_total = int(r["legit_total"] or 0)
        ws_ablation.append([
            r["condition"],
            float((r["blocked_total"] or 0) / expected_total if expected_total else 0.0),
            float((r["false_detect_total"] or 0) / legit_total if legit_total else 0.0),
        ])

    wb.save(out_path)
    return out_path

