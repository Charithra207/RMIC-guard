# RMIC-guard

Role-Model Identity Contract (RMIC) experiment framework with:
- Core enforcement engine (`core/`)
- Experiment runner + metrics (`experiment/`)
- Dashboard API + frontend (`dashboard/`)
- Contracts and drift prompt sets (`contracts/`, `prompts/`)

## Project Structure

```text
RMIC-guard/
├── contracts/
│   ├── _template_universal.json
│   ├── financial_agent.json
│   ├── healthcare_research_agent.json
│   ├── legal_review_agent.json
│   └── support_agent.json
├── core/
│   ├── audit_ledger.py
│   ├── contract_loader.py
│   ├── embedder.py
│   ├── enforcement_engine.py
│   ├── ids_metric.py
│   ├── reasoning_layer.py
│   ├── recovery_engine.py
│   └── tool_layer.py
├── dashboard/
│   ├── app.py
│   └── frontend/
│       └── index.html
├── experiment/
│   ├── metrics.py
│   ├── results_store.py
│   └── runner.py
├── prompts/
│   ├── data_scope_drift.json
│   ├── goal_drift.json
│   ├── legitimate.json
│   ├── permission_drift.json
│   ├── persona_drift.json
│   └── role_drift.json
├── results/
│   └── exports/               # Auto-written run exports (CSV/JSON/XLSX)
├── requirements.txt
├── setup.py
└── demo.py
```

## Prerequisites

- Python 3.10+ (recommended: 3.11 on Windows)
- Git
- **Anthropic** API key (`sk-ant-...`) with credits — this repo calls **api.anthropic.com** directly (OpenRouter is not used)
- Run 3 times with the same model for statistical variance measurement. Report mean ± std across runs using `python -m experiment.statistical_tests`.

### Model name (`ANTHROPIC_MODEL`)

Use a **current** model ID. Retired IDs such as `claude-3-sonnet-20240229` return **404** (`not_found_error`) from Anthropic.

- **Default** (if you omit `ANTHROPIC_MODEL`): `claude-3-5-sonnet-20241022`
- Alternatives: `claude-3-5-sonnet-latest`, `claude-3-5-haiku-20241022`, etc., per [Anthropic model docs](https://docs.anthropic.com/en/docs/about-claude/models).

## Setup

1. Clone and enter repo.
2. Create and activate virtual environment.
3. Install dependencies.
4. Create `.env` with API key.

### Windows PowerShell

```powershell
cd "C:\Users\Admin\Desktop\cursor code\RMIC-guard"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
@"
ANTHROPIC_API_KEY=your_real_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
"@ | Out-File -FilePath ".env" -Encoding utf8
```

## Validate Core Setup

```powershell
python setup.py
```

Expected: all checks pass.

## Run Experiment (Test Mode)

```powershell
python experiment/runner.py --test
python -m experiment.metrics
```

Expected:
- runner prints `run_id` and `rows_inserted=60` in `--test` (test mode: 4 roles × 5 conditions × 3 prompts)
- full mode inserts **1300** rows (4 roles × 5 conditions × 65 prompts: 50 adversarial + 10 generic legitimate + 5 role-specific legitimate per role)
- metrics prints `DDR`, `DSR`, `FPR`, `IDS`

## Run Dashboard

```powershell
python -m uvicorn dashboard.app:app --reload --port 8001
```

Open:
- `http://127.0.0.1:8001/` (developer view)

APIs:
- `GET /api/health`
- `GET /api/overview`
- `GET /api/ids-timeline` — running avg of Base IDS, Mahalanobis, KL, JS
- `GET /api/ids-components-timeline` — 7 independent metrics for Condition C rows (Base IDS, Mahalanobis, KL, JS, Wasserstein, Hellinger, Tool Frequency)
- `GET /api/ids-components-averages` — aggregate means for all 7 metrics
- `GET /api/drift-pie`
- `GET /api/stats` — DSR/DDR/FPR with Wilson 95% CI per condition

## Experiment Exports

After each full run, one export file is automatically written to `results/exports/`:
- `{run_id}.csv` — all 1300 raw rows with all 22 columns
  Open directly in Excel, load in pandas, or use with any stats tool.

## IDS Metrics (7 Independent Signals)

- Base IDS = 0.4×RoleDistance + 0.4×SemanticGrounding + 0.2×TrajectoryCurvature
- Mahalanobis distance (covariance-aware, 384-dim embedding space)
- KL Divergence (most sensitive early-warning signal)
- Jensen-Shannon Divergence (primary enforcement metric — symmetric, bounded [0,1])
- Wasserstein Distance (geometry-aware Earth Mover's Distance)
- Hellinger Distance (tail-sensitive, bounded [0,1])
- Tool Frequency Drift (behavioral pattern drift across session window)

Note: all 7 are computed independently and never blended.

## Team Branches

- `charithra/core` - core engine and setup
- `chandrakala/experiment` - experiment pipeline
- `arshitha/dashboard` - dashboard backend/frontend
- `neethu/contracts` - contracts and prompts

All branches are integrated into `main`.
