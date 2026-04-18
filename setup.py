"""
RMIC-Guard pre-flight validation (no API calls, no Anthropic credits).

Run from the repository root:
  python setup.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

CORE_FILES = (
    "contract_loader.py",
    "embedder.py",
    "ids_metric.py",
    "audit_ledger.py",
    "enforcement_engine.py",
    "reasoning_layer.py",
    "tool_layer.py",
    "recovery_engine.py",
)

CONTRACT_FILES = (
    "financial_agent.json",
    "support_agent.json",
    "healthcare_research_agent.json",
    "legal_review_agent.json",
)

# Retired on api.anthropic.com (404) — keep in sync with core/reasoning_layer.py
_RETIRED_ANTHROPIC_MODELS: frozenset[str] = frozenset(
    {
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-latest",
    }
)
_DEFAULT_MODEL = "claude-sonnet-4-6"


def main() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv()

    passed = 0
    failed = 0

    def ok_line(label: str) -> None:
        nonlocal passed
        print(f"[OK] {label}")
        passed += 1

    def warn_line(label: str, detail: str = "") -> None:
        nonlocal passed
        suffix = f" — {detail}" if detail else ""
        print(f"[WARN] {label}{suffix}")
        passed += 1

    def err_line(label: str, detail: str = "") -> None:
        nonlocal failed
        suffix = f" — {detail}" if detail else ""
        print(f"[ERROR] {label}{suffix}")
        failed += 1

    # 1 — Python version
    if sys.version_info >= (3, 10):
        ok_line("Python version is 3.10 or higher")
    else:
        err_line(
            "Python version is 3.10 or higher",
            f"found {sys.version_info.major}.{sys.version_info.minor}",
        )

    # 2 — .env and API key shape (no network)
    env_path = ROOT / ".env"
    if not env_path.is_file():
        err_line(".env and ANTHROPIC_API_KEY", f"missing {env_path}")
    else:
        an_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if an_key.startswith("sk-ant-"):
            ok_line(".env exists and ANTHROPIC_API_KEY starts with sk-ant-")
        else:
            err_line(
                ".env exists and ANTHROPIC_API_KEY starts with sk-ant-",
                "key missing, empty, or wrong prefix",
            )

    model_env = (os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if model_env in _RETIRED_ANTHROPIC_MODELS:
        warn_line(
            "ANTHROPIC_MODEL is retired (Anthropic API returns 404)",
            f"{model_env} — remove it or set ANTHROPIC_MODEL={_DEFAULT_MODEL}",
        )

    # 3 — core modules
    core_dir = ROOT / "core"
    missing_core = [f for f in CORE_FILES if not (core_dir / f).is_file()]
    if not missing_core:
        ok_line("All 8 core Python files are present")
    else:
        err_line("All 8 core Python files are present", f"missing: {', '.join(missing_core)}")

    # 4 — sealed contracts (recommended, but not required to run with verify_hash=False)
    contracts_dir = ROOT / "contracts"
    contract_errors: list[str] = []
    for name in CONTRACT_FILES:
        path = contracts_dir / name
        if not path.is_file():
            contract_errors.append(f"{name} (file missing)")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            contract_errors.append(f"{name} (invalid JSON: {exc})")
            continue
        if not isinstance(data, dict):
            contract_errors.append(f"{name} (root must be an object)")
            continue
        ch = data.get("contract_hash")
        if ch is None or (isinstance(ch, str) and ch.strip() == ""):
            contract_errors.append(f"{name} (contract_hash null or empty — seal required)")
        elif not isinstance(ch, str):
            contract_errors.append(f"{name} (contract_hash must be a string)")
    if not contract_errors:
        ok_line("All 4 contract files are sealed (contract_hash set)")
    else:
        warn_line(
            "Contracts are not sealed (contract_hash set)",
            "Runner can still execute with verify_hash=False; seal later on a compatible Python (3.11 recommended)",
        )

    # 5 — local embedding model (fastembed ONNX preferred, sentence-transformers fallback)
    sys.path.insert(0, str(ROOT))
    try:
        from core.embedder import embed_texts

        vec = embed_texts(["RMIC-Guard validation sentence."])
        shape = tuple(vec.shape)
        try:
            import fastembed  # noqa: F401

            backend = "fastembed (ONNX)"
        except ImportError:
            backend = "sentence-transformers (PyTorch)"
        if shape[0] == 1 and shape[1] > 0:
            ok_line(f"Embedding model loads via {backend}, shape {shape}")
        else:
            warn_line("Embedding model shape unexpected", f"got {shape}")
    except Exception as exc:  # noqa: BLE001 — surface any import/runtime failure
        warn_line("Embedding model loads", str(exc))

    # 6 — results/
    results_dir = ROOT / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        ok_line("results/ folder exists (created if missing)")
    except OSError as exc:
        err_line("results/ folder exists (created if missing)", str(exc))

    # Summary
    total = passed + failed
    print()
    print(f"Checks passed: {passed} / {total}")
    print(f"Checks failed: {failed} / {total}")
    print()

    if failed == 0:
        print("System ready. Run python demo.py to start.")
        sys.exit(0)
    print("Fix the errors above before running the experiment.")
    sys.exit(1)


if __name__ == "__main__":
    main()
