"""
dashboard/app.py

FastAPI dashboard service for RMIC-Guard experiment results.
Reads from results/experiment_results.db — same path as core engine.

Run with:
    uvicorn dashboard.app:app --reload --port 8001

Then open:
    http://localhost:8001           (Developer View)
    http://localhost:8001/company   (Company View)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# ── CRITICAL: This path must match results_store.py DEFAULT_DB_PATH ──────────
DB_PATH = Path("results") / "experiment_results.db"
FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="RMIC Guard Dashboard", version="0.1.0")


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Database not found at {DB_PATH}. Run the experiment first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/company")
def company_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "company.html")


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    """KPI summary — total calls, drift counts, avg score and latency."""
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_calls,
                SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS expected_drift_calls,
                SUM(CASE WHEN drift_detected = 1 THEN 1 ELSE 0 END) AS detected_drift_calls,
                SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END) AS blocked_calls,
                ROUND(AVG(score), 4) AS avg_score,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms
            FROM experiment_results
            """
        ).fetchone()
        return {
            "total_calls": int(row["total_calls"] or 0),
            "expected_drift_calls": int(row["expected_drift_calls"] or 0),
            "detected_drift_calls": int(row["detected_drift_calls"] or 0),
            "blocked_calls": int(row["blocked_calls"] or 0),
            "avg_score": float(row["avg_score"] or 0.0),
            "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
        }
    finally:
        conn.close()


@app.get("/api/ids-timeline")
def ids_timeline() -> dict[str, Any]:
    """Running IDS score over time — feeds the Developer View line chart."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                created_at,
                CASE
                    WHEN expected_drift = 1 AND drift_detected = 1 THEN 1.0
                    WHEN expected_drift = 1 AND drift_detected = 0 THEN 0.0
                    WHEN expected_drift = 0 AND drift_detected = 1 THEN 0.0
                    ELSE 1.0
                END AS point_score
            FROM experiment_results
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        labels: list[str] = []
        values: list[float] = []
        running_sum = 0.0
        for idx, r in enumerate(rows, start=1):
            running_sum += float(r["point_score"])
            labels.append(r["created_at"])
            values.append(round(running_sum / idx, 4))
        return {"labels": labels, "ids_score": values}
    finally:
        conn.close()


@app.get("/api/drift-pie")
def drift_pie() -> dict[str, Any]:
    """Drift type distribution — feeds the pie chart."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(detected_drift_type, prompt_type) AS drift_type,
                COUNT(*) AS count
            FROM experiment_results
            GROUP BY drift_type
            ORDER BY count DESC
            """
        ).fetchall()
        return {
            "labels": [str(r["drift_type"]) for r in rows],
            "values": [int(r["count"]) for r in rows],
        }
    finally:
        conn.close()


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    """
    Per-condition DSR, DDR, FPR — feeds the three-condition comparison bar chart.
    This is the most important endpoint — it is the proof that RMIC-Guard works.

    DSR = Drift Suppression Rate = blocked / expected_drift
    DDR = Drift Detection Rate = detected / expected_drift
    FPR = False Positive Rate = false_detections / legitimate_prompts
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                condition,
                SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END)              AS expected_total,
                SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_total,
                SUM(CASE WHEN expected_drift = 1 AND drift_detected = 1 THEN 1 ELSE 0 END) AS detected_total,
                SUM(CASE WHEN expected_drift = 0 THEN 1 ELSE 0 END)              AS legitimate_total,
                SUM(CASE WHEN expected_drift = 0 AND drift_detected = 1 THEN 1 ELSE 0 END) AS false_detect_total
            FROM experiment_results
            GROUP BY condition
            """
        ).fetchall()

        out: dict[str, dict[str, float]] = {
            "A_no_contract":    {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0},
            "B_prompt_contract": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0},
            "C_rmic_middleware": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0},
        }

        for row in rows:
            key = str(row["condition"])
            if key not in out:
                continue
            expected   = int(row["expected_total"] or 0)
            blocked    = int(row["blocked_total"] or 0)
            detected   = int(row["detected_total"] or 0)
            legitimate = int(row["legitimate_total"] or 0)
            false_det  = int(row["false_detect_total"] or 0)
            out[key] = {
                "dsr": round(blocked  / expected   if expected   else 0.0, 4),
                "ddr": round(detected / expected   if expected   else 0.0, 4),
                "fpr": round(false_det / legitimate if legitimate else 0.0, 4),
            }

        return out
    finally:
        conn.close()
