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
    """Running four-metric timeline over time for side-by-side comparison."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                created_at,
                COALESCE(ids_score, 0.0) AS ids_score,
                COALESCE(mahalanobis, mahalanobis_distance, 0.0) AS mahalanobis_distance,
                COALESCE(kl_divergence, 0.0) AS kl_divergence,
                COALESCE(js_divergence, 0.0) AS js_divergence
            FROM experiment_results
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        labels: list[str] = []
        ids_values: list[float] = []
        mahal_values: list[float] = []
        kl_values: list[float] = []
        js_values: list[float] = []
        ids_running = 0.0
        mahal_running = 0.0
        kl_running = 0.0
        js_running = 0.0
        for idx, r in enumerate(rows, start=1):
            ids_running += float(r["ids_score"])
            mahal_running += float(r["mahalanobis_distance"])
            kl_running += float(r["kl_divergence"])
            js_running += float(r["js_divergence"])
            labels.append(r["created_at"])
            ids_values.append(round(ids_running / idx, 4))
            mahal_values.append(round(mahal_running / idx, 4))
            kl_values.append(round(kl_running / idx, 4))
            js_values.append(round(js_running / idx, 4))
        return {
            "labels": labels,
            "ids_score": ids_values,
            "mahalanobis_distance": mahal_values,
            "kl_divergence": kl_values,
            "js_divergence": js_values,
        }
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


@app.get("/api/ids-components-timeline")
def ids_components_timeline() -> dict[str, Any]:
    """
    Per-row mixed IDS + statistical components (Condition C only, where computed).
    Used for multi-series charts (Mahalanobis, KL, JS, base IDS).
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                created_at,
                ids_score,
                base_ids,
                mahalanobis,
                kl_divergence,
                js_divergence
            FROM experiment_results
            WHERE condition = 'C_rmic_middleware'
              AND mahalanobis IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()

        def _f(val: Any) -> float | None:
            if val is None:
                return None
            return float(val)

        return {
            "labels": [str(r["created_at"]) for r in rows],
            "mixed_ids": [_f(r["ids_score"]) for r in rows],
            "base_ids": [_f(r["base_ids"]) for r in rows],
            "mahalanobis": [_f(r["mahalanobis"]) for r in rows],
            "kl_divergence": [_f(r["kl_divergence"]) for r in rows],
            "js_divergence": [_f(r["js_divergence"]) for r in rows],
        }
    finally:
        conn.close()


@app.get("/api/ids-components-averages")
def ids_components_averages() -> dict[str, Any]:
    """Aggregate means over Condition C rows that have component scores stored."""
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT
                ROUND(AVG(mahalanobis), 4) AS avg_mahalanobis,
                ROUND(AVG(kl_divergence), 4) AS avg_kl,
                ROUND(AVG(js_divergence), 4) AS avg_js,
                ROUND(AVG(base_ids), 4) AS avg_base_ids,
                ROUND(AVG(ids_score), 4) AS avg_mixed_ids,
                COUNT(*) AS n
            FROM experiment_results
            WHERE condition = 'C_rmic_middleware'
              AND mahalanobis IS NOT NULL
            """
        ).fetchone()
        return {
            "avg_mahalanobis": float(row["avg_mahalanobis"] or 0.0),
            "avg_kl": float(row["avg_kl"] or 0.0),
            "avg_js": float(row["avg_js"] or 0.0),
            "avg_base_ids": float(row["avg_base_ids"] or 0.0),
            "avg_mixed_ids": float(row["avg_mixed_ids"] or 0.0),
            "sample_count": int(row["n"] or 0),
        }
    finally:
        conn.close()
