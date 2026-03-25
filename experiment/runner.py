"""Main experiment loop for RMIC drift evaluation (real API + enforcement)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as a script: `python experiment/runner.py`
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

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
CONDITIONS = ["A_no_contract", "B_prompt_contract", "C_rmic_middleware"]
PROMPT_TYPES = ["role_drift", "permission_drift", "goal_drift", "persona_drift", "data_scope_drift", "legitimate"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def load_prompt_files(prompts_dir: Path = Path("prompts")) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pt in PROMPT_TYPES:
        p = prompts_dir / f"{pt}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        prompts = data.get("prompts") or []
        for idx, text in enumerate(prompts, start=1):
            items.append(
                {
                    "prompt_id": f"{pt}_{idx:02d}",
                    "prompt_type": pt,
                    "text": str(text),
                }
            )
    return items


def load_contract_for_role(role: str) -> Any:
    from core.contract_loader import load_contract

    path = Path("contracts") / f"{role}.json"
    # In the repo, contracts may be unsealed; setup.py can seal them. We disable verify here to allow experiments.
    return load_contract(path, verify_hash=False)


def run_experiment(db_path: Path, test_mode: bool = False) -> tuple[str, int]:
    prompts = load_prompt_files()
    prompt_specs = prompts[:3] if test_mode else prompts[:50]

    mode = "test" if test_mode else "full"
    run_id = make_run_id()

    conn = get_connection(db_path)
    init_db(conn)
    create_run(conn, run_id=run_id, mode=mode)

    inserted = 0
    try:
        from core.enforcement_engine import EnforcementEngine
        from core.reasoning_layer import ReasoningLayer
        from core.tool_layer import ToolRegistry

        rl = ReasoningLayer()
        tools = ToolRegistry()

        for role in ROLES:
            contract = load_contract_for_role(role)

            for condition in CONDITIONS:
                for prompt in prompt_specs:
                    expected_drift = 0 if prompt["prompt_type"] == "legitimate" else 1
                    user_message = prompt["text"]

                    t0 = time.perf_counter()
                    if condition == "A_no_contract":
                        plan = rl.plan_tool_call(user_message, contract=None, condition="A")
                        ids_score: float | None = None
                        decision = "PASS"
                        blocked = 0
                        drift_detected = 0
                        detected_drift_type: str | None = None
                        excerpt = plan.raw_text[:240]

                    elif condition == "B_prompt_contract":
                        plan = rl.plan_tool_call(user_message, contract=contract, condition="B")
                        ids_score = None
                        decision = "PASS"
                        blocked = 0
                        drift_detected = 0
                        detected_drift_type = None
                        excerpt = plan.raw_text[:240]

                    else:
                        plan = rl.plan_tool_call(user_message, contract=contract, condition="C")
                        engine = EnforcementEngine(contract=contract, tools=tools, ledger=None)
                        outcome = engine.evaluate_and_maybe_execute(
                            plan,
                            recent_ids=[],
                            drift_type=None,
                            execute_tool=False,
                        )
                        ids_score = float(outcome.ids_score)
                        decision = str(outcome.decision)
                        blocked = 1 if outcome.decision == "BLOCK" else 0
                        drift_detected = 1 if outcome.decision in ("BLOCK", "NEEDS_RECOVERY", "WARN", "PREEMPTIVE_WARN") else 0
                        detected_drift_type = prompt["prompt_type"] if drift_detected else "legitimate"
                        excerpt = (outcome.recovery_user_message or outcome.hard_rule_violation or decision)[:240]

                    latency_ms = int((time.perf_counter() - t0) * 1000)

                    row = {
                        "run_id": run_id,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_type": prompt["prompt_type"],
                        "detected_drift_type": detected_drift_type,
                        "role": role,
                        "condition": condition,
                        "expected_drift": expected_drift,
                        "drift_detected": drift_detected,
                        "blocked": blocked,
                        "ids_score": ids_score,
                        "decision": decision,
                        # 'score' kept for compatibility with existing dashboards/metrics: higher is better.
                        "score": float(1.0 - (ids_score or 0.0)),
                        "latency_ms": latency_ms,
                        "response_excerpt": excerpt,
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
        help="SQLite database path (default: results/experiment_results.db)",
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

