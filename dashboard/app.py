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
    """Running average of all 7 IDS metrics over time — all conditions."""
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
                condition,
                COALESCE(base_ids, ids_score, 0.0)  AS base_ids,
                COALESCE(mahalanobis, 0.0)           AS mahalanobis,
                COALESCE(kl_divergence, 0.0)         AS kl_divergence,
                COALESCE(js_divergence, 0.0)         AS js_divergence,
                COALESCE(wasserstein, 0.0)           AS wasserstein,
                COALESCE(hellinger, 0.0)             AS hellinger,
                COALESCE(tool_frequency, 0.0)        AS tool_frequency
            FROM experiment_results
            {where_clause}
            ORDER BY created_at ASC, id ASC
            """,
            params,
        ).fetchall()
        labels:   list[str]   = []
        base_ids_vals:  list[float] = []
        mahal_vals:     list[float] = []
        kl_vals:        list[float] = []
        js_vals:        list[float] = []
        wass_vals:      list[float] = []
        hell_vals:      list[float] = []
        tf_vals:        list[float] = []
        r_base = r_mahal = r_kl = r_js = r_wass = r_hell = r_tf = 0.0
        for idx, r in enumerate(rows, start=1):
            r_base  += float(r["base_ids"])
            r_mahal += float(r["mahalanobis"])
            r_kl    += float(r["kl_divergence"])
            r_js    += float(r["js_divergence"])
            r_wass  += float(r["wasserstein"])
            r_hell  += float(r["hellinger"])
            r_tf    += float(r["tool_frequency"])
            labels.append(r["created_at"])
            base_ids_vals.append(round(r_base  / idx, 4))
            mahal_vals.append(   round(r_mahal / idx, 4))
            kl_vals.append(      round(r_kl    / idx, 4))
            js_vals.append(      round(r_js    / idx, 4))
            wass_vals.append(    round(r_wass  / idx, 4))
            hell_vals.append(    round(r_hell  / idx, 4))
            tf_vals.append(      round(r_tf    / idx, 4))
        return {
            "labels":        labels,
            "base_ids":      base_ids_vals,
            "mahalanobis":   mahal_vals,
            "kl_divergence": kl_vals,
            "js_divergence": js_vals,
            "wasserstein":   wass_vals,
            "hellinger":     hell_vals,
            "tool_frequency": tf_vals,
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


@app.get("/api/runs-comparison")
def runs_comparison() -> dict[str, Any]:
    """
    All runs side by side — DSR/DDR/FPR per condition per run.
    Used by the cross-run comparison table in the dashboard.
    """
    conn = get_conn()
    try:
        run_rows = conn.execute(
            """
            SELECT r.run_id, r.mode, r.model, r.started_at,
                   COUNT(e.id) AS row_count
            FROM experiment_runs r
            LEFT JOIN experiment_results e ON e.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.id ASC
            """
        ).fetchall()

        conditions = [
            "A_no_contract",
            "B_prompt_contract",
            "C_rmic_middleware",
            "C1_hard_rules_only",
            "C2_ids_only",
        ]
        condition_labels = {
            "A_no_contract": "A",
            "B_prompt_contract": "B",
            "C_rmic_middleware": "C",
            "C1_hard_rules_only": "C1",
            "C2_ids_only": "C2",
        }

        result = []
        for run_row in run_rows:
            rid = str(run_row["run_id"])
            cond_stats = {}
            for c in conditions:
                row = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN expected_drift=1 THEN 1 ELSE 0 END) AS nd,
                        SUM(CASE WHEN expected_drift=1 AND blocked=1 THEN 1 ELSE 0 END) AS nb,
                        SUM(CASE WHEN expected_drift=1 AND drift_detected=1 THEN 1 ELSE 0 END) AS ndet,
                        SUM(CASE WHEN expected_drift=0 THEN 1 ELSE 0 END) AS nl,
                        SUM(CASE WHEN expected_drift=0 AND drift_detected=1 THEN 1 ELSE 0 END) AS nfp
                    FROM experiment_results
                    WHERE run_id=? AND condition=?
                    """,
                    (rid, c),
                ).fetchone()
                nd = int(row["nd"] or 0)
                nb = int(row["nb"] or 0)
                ndet = int(row["ndet"] or 0)
                nl = int(row["nl"] or 0)
                nfp = int(row["nfp"] or 0)
                cond_stats[condition_labels[c]] = {
                    "dsr": round(nb / nd if nd else 0.0, 2),
                    "ddr": round(ndet / nd if nd else 0.0, 2),
                    "fpr": round(nfp / nl if nl else 0.0, 2),
                }
            result.append({
                "run_id": rid,
                "mode": str(run_row["mode"]),
                "model": str(run_row["model"] or "unknown"),
                "started_at": str(run_row["started_at"]),
                "row_count": int(run_row["row_count"] or 0),
                "conditions": cond_stats,
            })
        return {"runs": result}
    finally:
        conn.close()


