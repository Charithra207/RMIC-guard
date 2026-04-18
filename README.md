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
