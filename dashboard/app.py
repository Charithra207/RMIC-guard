"""
dashboard/app.py

FastAPI dashboard service for RMIC-Guard experiment results.
Reads from results/experiment_results.db — same path as core engine.

Run with:
    uvicorn dashboard.app:app --reload --port 8001

Then open:
    http://localhost:8001           (Developer View)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
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


def get_latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT run_id FROM experiment_runs
        WHERE EXISTS (
            SELECT 1 FROM experiment_results e WHERE e.run_id = experiment_runs.run_id
        )
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    return str(row["run_id"]) if row else None


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    """List all experiment runs for the run selector."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT r.run_id, r.mode, r.model, r.started_at,
                   COUNT(e.id) AS row_count
            FROM experiment_runs r
            LEFT JOIN experiment_results e ON e.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.id DESC
            """
        ).fetchall()
        return {
            "runs": [
                {
                    "run_id": str(r["run_id"]),
                    "mode": str(r["mode"]),
                    "model": str(r["model"] or "unknown"),
                    "started_at": str(r["started_at"]),
                    "row_count": int(r["row_count"] or 0),
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


@app.get("/api/overview")
def overview(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """KPI summary — total calls, drift counts, avg score and latency."""
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_calls,
                SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS expected_drift_calls,
                SUM(CASE WHEN drift_detected = 1 THEN 1 ELSE 0 END) AS detected_drift_calls,
                SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END) AS blocked_calls,
                ROUND(AVG(score), 4) AS avg_score,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms
            FROM experiment_results
            {where_clause}
            """,
            params,
        ).fetchone()
        return {
            "total_calls": int(row["total_calls"] or 0),
            "expected_drift_calls": int(row["expected_drift_calls"] or 0),
            "detected_drift_calls": int(row["detected_drift_calls"] or 0),
            "blocked_calls": int(row["blocked_calls"] or 0),
            "avg_score": float(row["avg_score"] or 0.0),
            "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
            "active_run_id": run_id,
        }
    finally:
        conn.close()


@app.get("/api/ids-timeline")
def ids_timeline(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """Running four-metric timeline over time for side-by-side comparison."""
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        rows = conn.execute(
            f"""
            SELECT
                created_at,
                COALESCE(ids_score, 0.0) AS ids_score,
                COALESCE(mahalanobis, 0.0) AS mahalanobis_distance,
                COALESCE(kl_divergence, 0.0) AS kl_divergence,
                COALESCE(js_divergence, 0.0) AS js_divergence
            FROM experiment_results
            {where_clause}
            ORDER BY created_at ASC, id ASC
            """,
            params,
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
            "ids_label": "Base IDS",
            "mahalanobis_distance": mahal_values,
            "kl_divergence": kl_values,
            "js_divergence": js_values,
            "active_run_id": run_id,
        }
    finally:
        conn.close()


@app.get("/api/drift-pie")
def drift_pie(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """Drift type distribution — feeds the pie chart."""
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(detected_drift_type, prompt_type) AS drift_type,
                COUNT(*) AS count
            FROM experiment_results
            {where_clause}
            GROUP BY drift_type
            ORDER BY count DESC
            """,
            params,
        ).fetchall()
        return {
            "labels": [str(r["drift_type"]) for r in rows],
            "values": [int(r["count"]) for r in rows],
            "active_run_id": run_id,
        }
    finally:
        conn.close()


@app.get("/api/stats")
def stats(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """
    Per-condition DSR, DDR, FPR — feeds the three-condition comparison bar chart.
    This is the most important endpoint — it is the proof that RMIC-Guard works.

    DSR = Drift Suppression Rate = blocked / expected_drift
    DDR = Drift Detection Rate = detected / expected_drift
    FPR = False Positive Rate = false_detections / legitimate_prompts
    """
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        rows = conn.execute(
            f"""
            SELECT
                condition,
                SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END)              AS expected_total,
                SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_total,
                SUM(CASE WHEN expected_drift = 1 AND drift_detected = 1 THEN 1 ELSE 0 END) AS detected_total,
                SUM(CASE WHEN expected_drift = 0 THEN 1 ELSE 0 END)              AS legitimate_total,
                SUM(CASE WHEN expected_drift = 0 AND drift_detected = 1 THEN 1 ELSE 0 END) AS false_detect_total
            FROM experiment_results
            {where_clause}
            GROUP BY condition
            """,
            params,
        ).fetchall()

        out: dict[str, dict[str, float | list[float]]] = {
            "A_no_contract": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0, "dsr_ci": [0.0, 0.0]},
            "B_prompt_contract": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0, "dsr_ci": [0.0, 0.0]},
            "C_rmic_middleware": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0, "dsr_ci": [0.0, 0.0]},
            "C1_hard_rules_only": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0, "dsr_ci": [0.0, 0.0]},
            "C2_ids_only": {"dsr": 0.0, "ddr": 0.0, "fpr": 0.0, "dsr_ci": [0.0, 0.0]},
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
            p = blocked / expected if expected else 0.0
            z = 1.96
            z2 = z * z
            denom = 1.0 + (z2 / expected) if expected else 1.0
            center = ((p + (z2 / (2.0 * expected))) / denom) if expected else 0.0
            margin = (
                z * (((p * (1.0 - p) + z2 / (4.0 * expected)) / expected) ** 0.5) / denom
                if expected else 0.0
            )
            out[key] = {
                "dsr": round(blocked  / expected   if expected   else 0.0, 4),
                "ddr": round(detected / expected   if expected   else 0.0, 4),
                "fpr": round(false_det / legitimate if legitimate else 0.0, 4),
                "dsr_ci": [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)],
            }

        return {"active_run_id": run_id, "stats": out}
    finally:
        conn.close()


@app.get("/api/ids-components-timeline")
def ids_components_timeline(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """
    Per-row base IDS + Mahalanobis / KL / JS (Condition C rows where computed;
    independent metrics from the runner, not a single mixed IDS score).
    """
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE condition = 'C_rmic_middleware' AND mahalanobis IS NOT NULL"
        params: tuple[str, ...] = ()
        if run_id:
            where_clause += " AND run_id = ?"
            params = (run_id,)
        rows = conn.execute(
            f"""
            SELECT
                created_at,
                ids_score,
                base_ids,
                mahalanobis,
                kl_divergence,
                js_divergence,
                wasserstein,
                hellinger,
                tool_frequency
            FROM experiment_results
            {where_clause}
            ORDER BY id ASC
            """,
            params,
        ).fetchall()

        def _f(val: Any) -> float | None:
            if val is None:
                return None
            return float(val)

        return {
            "labels": [str(r["created_at"]) for r in rows],
            "base_ids": [_f(r["base_ids"]) for r in rows],
            "mahalanobis": [_f(r["mahalanobis"]) for r in rows],
            "kl_divergence": [_f(r["kl_divergence"]) for r in rows],
            "js_divergence": [_f(r["js_divergence"]) for r in rows],
            "wasserstein": [_f(r["wasserstein"]) for r in rows],
            "hellinger": [_f(r["hellinger"]) for r in rows],
            "tool_frequency": [_f(r["tool_frequency"]) for r in rows],
            "active_run_id": run_id,
        }
    finally:
        conn.close()


@app.get("/api/ids-components-averages")
def ids_components_averages(run_id: str | None = Query(default=None)) -> dict[str, Any]:
    """Aggregate means over Condition C rows that have component scores stored."""
    conn = get_conn()
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        where_clause = "WHERE condition = 'C_rmic_middleware' AND mahalanobis IS NOT NULL"
        params: tuple[str, ...] = ()
        if run_id:
            where_clause += " AND run_id = ?"
            params = (run_id,)
        row = conn.execute(
            f"""
            SELECT
                ROUND(AVG(mahalanobis), 4) AS avg_mahalanobis,
                ROUND(AVG(kl_divergence), 4) AS avg_kl,
                ROUND(AVG(js_divergence), 4) AS avg_js,
                ROUND(AVG(wasserstein), 4) AS avg_wasserstein,
                ROUND(AVG(hellinger), 4) AS avg_hellinger,
                ROUND(AVG(tool_frequency), 4) AS avg_tool_frequency,
                ROUND(AVG(base_ids), 4) AS avg_base_ids,
                COUNT(*) AS n
            FROM experiment_results
            {where_clause}
            """,
            params,
        ).fetchone()
        return {
            "avg_mahalanobis": float(row["avg_mahalanobis"] or 0.0),
            "avg_kl": float(row["avg_kl"] or 0.0),
            "avg_js": float(row["avg_js"] or 0.0),
            "avg_wasserstein": float(row["avg_wasserstein"] or 0.0),
            "avg_hellinger": float(row["avg_hellinger"] or 0.0),
            "avg_tool_frequency": float(row["avg_tool_frequency"] or 0.0),
            "avg_base_ids": float(row["avg_base_ids"] or 0.0),
            "sample_count": int(row["n"] or 0),
            "active_run_id": run_id,
        }
    finally:
        conn.close()
