"""Main experiment loop for drift evaluation."""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from experiment.results_store import (
        DEFAULT_DB_PATH,
        complete_run,
        create_run,
        get_connection,
        init_db,
        insert_result,
    )
except ModuleNotFoundError:
    from results_store import (  # type: ignore
        DEFAULT_DB_PATH,
        complete_run,
        create_run,
        get_connection,
        init_db,
        insert_result,
    )


ROLES = ["financial_agent", "support_agent", "healthcare_research_agent"]
CONDITIONS = ["control", "light_guard", "strict_guard"]
PROMPT_TYPES = [
    "role_drift",
    "permission_drift",
    "goal_drift",
    "persona_drift",
    "data_scope_drift",
    "legitimate",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def deterministic_rng(seed_key: str) -> random.Random:
    seed_hex = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(seed_hex, 16))


def simulate_model_call(role: str, condition: str, prompt_id: str, prompt_type: str) -> dict[str, int | float | str]:
    rng = deterministic_rng(f"{role}|{condition}|{prompt_id}|{prompt_type}")
    expected_drift = 0 if prompt_type == "legitimate" else 1

    detect_base = {"control": 0.40, "light_guard": 0.65, "strict_guard": 0.82}[condition]
    fpr_base = {"control": 0.16, "light_guard": 0.10, "strict_guard": 0.06}[condition]
    score = max(0.0, min(1.0, rng.gauss(0.70, 0.15)))
    latency_ms = int(max(100, rng.gauss(420, 140)))

    if expected_drift:
        drift_detected = 1 if rng.random() < detect_base else 0
        blocked = drift_detected
    else:
        drift_detected = 1 if rng.random() < fpr_base else 0
        blocked = drift_detected

    excerpt = (
        "Request denied due to policy drift risk."
        if blocked
        else "Request processed successfully."
    )
    return {
        "expected_drift": expected_drift,
        "drift_detected": drift_detected,
        "blocked": blocked,
        "score": round(score, 4),
        "latency_ms": latency_ms,
        "response_excerpt": excerpt,
    }


def build_prompt_specs(total_prompts: int) -> list[dict[str, str]]:
    prompt_specs: list[dict[str, str]] = []
    for i in range(1, total_prompts + 1):
        prompt_type = PROMPT_TYPES[(i - 1) % len(PROMPT_TYPES)]
        prompt_specs.append({"prompt_id": f"p{i:03d}", "prompt_type": prompt_type})
    return prompt_specs


def run_experiment(db_path: Path, test_mode: bool = False) -> tuple[str, int]:
    prompt_count = 3 if test_mode else 50
    prompt_specs = build_prompt_specs(prompt_count)
    mode = "test" if test_mode else "full"
    run_id = make_run_id()

    conn = get_connection(db_path)
    init_db(conn)
    create_run(conn, run_id=run_id, mode=mode)

    inserted = 0
    try:
        for role in ROLES:
            for condition in CONDITIONS:
                for prompt in prompt_specs:
                    simulated = simulate_model_call(
                        role=role,
                        condition=condition,
                        prompt_id=prompt["prompt_id"],
                        prompt_type=prompt["prompt_type"],
                    )
                    row = {
                        "run_id": run_id,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_type": prompt["prompt_type"],
                        "role": role,
                        "condition": condition,
                        "expected_drift": simulated["expected_drift"],
                        "drift_detected": simulated["drift_detected"],
                        "blocked": simulated["blocked"],
                        "score": simulated["score"],
                        "latency_ms": simulated["latency_ms"],
                        "response_excerpt": simulated["response_excerpt"],
                        "created_at": utc_now_iso(),
                    }
                    insert_result(conn, row)
                    inserted += 1
                    if test_mode:
                        time.sleep(0.01)
        conn.commit()
    finally:
        complete_run(conn, run_id=run_id)
        conn.close()

    return run_id, inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMIC drift experiment.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database path (default: data/results.db)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run lightweight test mode (3 prompts instead of 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id, inserted = run_experiment(db_path=args.db_path, test_mode=args.test)
    print(f"run_id={run_id}")
    print(f"rows_inserted={inserted}")
    print(f"db_path={args.db_path}")


if __name__ == "__main__":
    main()

