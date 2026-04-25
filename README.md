# RMIC-Guard

**Role-Model Identity Contract Guard** — research framework proving that enforcement
position (external middleware vs. in-prompt self-policing) is the primary determinant
of role identity drift suppression in autonomous LLM agents.

---

## Prerequisites

- Python 3.11 (Windows recommended)
- Anthropic API key (`sk-ant-...`) with credits

---

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root:

```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

---

## Validate

```powershell
python setup.py
```

Expected: 6/6 checks pass.

---

## Run Experiment

**Test first** (60 API calls, verify setup works):

```powershell
python -m experiment.runner --test
```

Expected output: `rows_inserted=60`

**Full run** (1,300 API calls per run, ~$8–15 at Sonnet 4.6 pricing):

```powershell
python -m experiment.runner
```

Run 3 times for statistical variance.

---

## Statistical Report

```powershell
python -m experiment.statistical_tests
```

Requires 2+ full runs. Report saved to `results/statistical_report.txt`.

---

## Dashboard

```powershell
python -m uvicorn dashboard.app:app --reload --port 8001
```

Open `http://127.0.0.1:8001` — use the run selector to switch between runs.

---

## Experimental Design

**4 Roles:** `financial_agent`, `support_agent`, `healthcare_research_agent`, `legal_review_agent`

**5 Conditions:**

| Condition | Description |
|---|---|
| `A_no_contract` | Baseline — no contract, no enforcement |
| `B_prompt_contract` | Contract in system prompt — LLM self-polices |
| `C_rmic_middleware` | External middleware — full dual-pass enforcement |
| `C1_hard_rules_only` | Ablation — hard rules only, no IDS |
| `C2_ids_only` | Ablation — semantic IDS only, no hard rules |

**65 prompts per role per condition:** 50 adversarial + 10 generic legitimate + 5 role-specific legitimate

---

## IDS Metrics (7 Independent Signals)

| Metric | Description |
|---|---|
| Base IDS | 0.4 × RoleDistance + 0.4 × SemanticGrounding + 0.2 × TrajectoryCurvature |
| Mahalanobis | Covariance-aware distance in 384-dim embedding space |
| KL Divergence | Asymmetric early-warning divergence signal |
| Jensen–Shannon | Primary enforcement metric — symmetric, bounded [0, 1] |
| Wasserstein | Geometry-aware Earth Mover's Distance |
| Hellinger | Tail-sensitive, bounded [0, 1] |
| Tool Frequency | Behavioral pattern drift across session window |

All 7 computed independently. Never blended.

---

## Key Metrics

| Metric | Definition |
|---|---|
| DSR | Drift Suppression Rate = blocked / expected_drift |
| DDR | Drift Detection Rate = detected / expected_drift |
| FPR | False Positive Rate = false_detections / legitimate |

---

## Exports

Each full run auto-exports to `results/exports/{run_id}.csv` (1,300 rows, 22 columns).

---

## Team Branches

| Branch | Scope |
|---|---|
| `charithra/core` | Core engine |
| `chandrakala/experiment` | Experiment pipeline |
| `arshitha/dashboard` | Dashboard |
| `neethu/contracts` | Contracts and prompts |
