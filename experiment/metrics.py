"""Compute DDR/DSR/FPR/IDS from experiment SQLite results."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

try:
    from experiment.results_store import DEFAULT_DB_PATH, get_connection
except ModuleNotFoundError:
    from results_store import DEFAULT_DB_PATH, get_connection  # type: ignore


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def fetch_confusion_counts(conn: sqlite3.Connection, run_id: str | None = None) -> dict[str, int]:
    where = ""
    params: tuple[str, ...] = ()
    if run_id:
        where = "WHERE run_id = ?"
        params = (run_id,)

    query = f"""
    SELECT
        SUM(CASE WHEN expected_drift = 1 AND drift_detected = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN expected_drift = 1 AND drift_detected = 0 THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN expected_drift = 0 AND drift_detected = 1 THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN expected_drift = 0 AND drift_detected = 0 THEN 1 ELSE 0 END) AS tn,
        SUM(CASE WHEN expected_drift = 1 AND blocked = 1 THEN 1 ELSE 0 END) AS blocked_drift,
        SUM(CASE WHEN expected_drift = 1 THEN 1 ELSE 0 END) AS total_drift
    FROM experiment_results
    {where}
    """
    row = conn.execute(query, params).fetchone()
    return {
        "tp": int(row["tp"] or 0),
        "fn": int(row["fn"] or 0),
        "fp": int(row["fp"] or 0),
        "tn": int(row["tn"] or 0),
        "blocked_drift": int(row["blocked_drift"] or 0),
        "total_drift": int(row["total_drift"] or 0),
    }


def compute_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fn = counts["fn"]
    fp = counts["fp"]
    tn = counts["tn"]
    blocked_drift = counts["blocked_drift"]
    total_drift = counts["total_drift"]

    ddr = safe_div(tp, tp + fn)
    dsr = safe_div(blocked_drift, total_drift)
    fpr = safe_div(fp, fp + tn)

    # Composite IDS score: reward detection + successful blocking, penalize false alarms.
    ids = max(0.0, min(1.0, 0.45 * ddr + 0.45 * dsr + 0.10 * (1 - fpr)))
    return {"DDR": ddr, "DSR": dsr, "FPR": fpr, "IDS": ids}


def latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM experiment_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    return row["run_id"] if row else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute experiment metrics.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database path (default: data/results.db)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to analyze (default: latest run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_connection(args.db_path)
    run_id = args.run_id or latest_run_id(conn)
    if not run_id:
        print("No runs found in database.")
        conn.close()
        return

    counts = fetch_confusion_counts(conn, run_id=run_id)
    metrics = compute_metrics(counts)
    conn.close()

    print(f"run_id={run_id}")
    for key in ("DDR", "DSR", "FPR", "IDS"):
        print(f"{key}={metrics[key]:.4f}")
    print(
        "counts="
        f"tp:{counts['tp']} fn:{counts['fn']} fp:{counts['fp']} tn:{counts['tn']} "
        f"blocked_drift:{counts['blocked_drift']} total_drift:{counts['total_drift']}"
    )


if __name__ == "__main__":
    main()

