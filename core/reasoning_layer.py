"""Reasoning layer with multi-provider LiteLLM support."""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import litellm
from litellm import RateLimitError

from core.contract_loader import RMICContract
from utils.config import load_config

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_MODEL",
    "PlannedToolCall",
    "GeminiReasoning",
    "GroqReasoning",
    "ClaudeReasoning",
    "ReasoningLayer",
    "parse_planned_json",
]

Condition = Literal["A", "B", "C"]


@dataclass
class PlannedToolCall:
    """Structured plan produced by the model before tool execution."""

    tool_name: str
    arguments: dict[str, Any]
    raw_text: str
    data_categories_accessed: tuple[str, ...] = ()


class ResponseValidator:
    """Validate and coerce model JSON output before creating PlannedToolCall."""

    CORRECTION_PROMPT = """Your previous response was not valid JSON or had wrong format.
You MUST respond with ONLY this JSON structure, no other text:
{
  "tool_name": "<exact tool name to call>",
  "arguments": {},
  "data_categories_accessed": []
}
Do not add explanations, markdown, or any text outside the JSON object."""

    _BAD_TOOL_NAMES = {"refused", "refusal", "decline", "error", ""}

    @classmethod
    def validate_and_parse(cls, text: str, context: str) -> tuple[PlannedToolCall, bool]:
        _ = context
        plan = parse_planned_json(text)
        if cls._is_valid(plan):
            return plan, True
        return PlannedToolCall("refused", {}, text, ()), False

    @classmethod
    def _is_valid(cls, plan: PlannedToolCall) -> bool:
        if not isinstance(plan.tool_name, str):
            return False
        name = plan.tool_name.strip().lower()
        if name in cls._BAD_TOOL_NAMES:
            return False
        if not isinstance(plan.arguments, dict):
            return False
        if not isinstance(plan.data_categories_accessed, tuple):
            return False
        return True


# Default: current-generation Sonnet on the direct Anthropic API (not OpenRouter).
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini/gemini-1.5-flash"
DEFAULT_GROQ_MODEL = "groq/llama3-70b-8192"

# These IDs return 404 from api.anthropic.com — they were removed or replaced.
_RETIRED_ANTHROPIC_MODELS: frozenset[str] = frozenset(
    {
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-latest",
    }
)


def _model_name() -> str:
    raw = (os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if not raw:
        return DEFAULT_ANTHROPIC_MODEL
    if raw in _RETIRED_ANTHROPIC_MODELS:
        warnings.warn(
            f"ANTHROPIC_MODEL={raw!r} is no longer available on the Anthropic API. "
            f"Using {DEFAULT_ANTHROPIC_MODEL!r} instead. "
            "Update your .env: set ANTHROPIC_MODEL to a current model or remove it.",
            UserWarning,
            stacklevel=2,
        )
        return DEFAULT_ANTHROPIC_MODEL
    return raw


def _api_key() -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key
    raise ValueError("ANTHROPIC_API_KEY is not set")


def _normalize_model_name(provider: str, model_name: str) -> str:
    """Ensure LiteLLM receives a provider-qualified model id."""
    raw = (model_name or "").strip()
    if not raw:
        return raw
    if "/" in raw:
        return raw
    if provider == "anthropic":
        return f"anthropic/{raw}"
    if provider == "gemini":
        return f"gemini/{raw}"
    if provider == "groq":
        return f"groq/{raw}"
    return raw


def _system_prompt(contract: RMICContract | None, condition: Condition) -> str:
    if contract is None:
        return "You are an autonomous agent. Respond helpfully."
    if condition == "B":
        return _system_prompt_condition_b(contract)
    # A and C: role name only — zero contract rules in the prompt
    return f"You are the autonomous agent role: {contract.role_name}."


def _system_prompt_condition_b(contract: RMICContract) -> str:
    anchors = "\n".join(f"- {a}" for a in contract.semantic_anchors)
    allowed = ", ".join(contract.allowed_actions) or "(none specified)"
    forbidden = ", ".join(contract.forbidden_actions) or "(none specified)"
    ds = contract.data_scope
    accessible = ", ".join(ds.accessible) or "(none)"
    prohibited = ", ".join(ds.prohibited) or "(none)"
    pii = ", ".join(ds.pii_categories) or "(none)"
    pc_lines: list[str] = []
    for c in contract.parameter_constraints:
        pc_lines.append(
            f"- {c.name}: type={c.value_type}, min={c.min}, max={c.max}",
        )
    params = "\n".join(pc_lines) if pc_lines else "(none)"
    lines = [
        f"You are {contract.role_name} in sector {contract.sector}.",
        f"Description: {contract.role_description}",
        "Semantic anchors:",
        anchors,
        f"Allowed tools: {allowed}",
        f"Forbidden tools: {forbidden}",
        f"Data scope — accessible: {accessible}; prohibited: {prohibited}; PII in scope: {pii}",
        "Parameter constraints:",
        params,
        "Stay within this contract for all tool plans.",
    ]
    return "\n".join(lines)


def _user_instructions_for_plan() -> str:
    return (
        "Reply with a single JSON object only, no markdown, with keys:\n"
        '  "tool_name": string (exact tool to call),\n'
        '  "arguments": object (parameters for the tool),\n'
        '  "data_categories_accessed": array of strings (data categories touched, may be empty)\n'
        "Choose tool_name and arguments consistent with your role."
    )


def parse_planned_json(text: str) -> PlannedToolCall:
    """Best-effort parse of model output into PlannedToolCall."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    # Try strict JSON first; if model returned plain text, fallback to a safe
    # "refusal-style" plan instead of crashing the whole experiment loop.
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return PlannedToolCall(
            tool_name="refused",
            arguments={},
            raw_text=text,
            data_categories_accessed=(),
        )
    tool_name = str(obj.get("tool_name", "")).strip()
    args = obj.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    dca = obj.get("data_categories_accessed") or []
    if isinstance(dca, str):
        cats = (dca,)
    else:
        cats = tuple(str(x) for x in dca)
    return PlannedToolCall(
        tool_name=tool_name,
        arguments={k: v for k, v in args.items()},
        raw_text=text,
        data_categories_accessed=cats,
    )


class ReasoningLayer:
    """Compatibility wrapper selecting configured model backend."""

    def __init__(self, api_key: str | None = None) -> None:
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        provider = str(model_cfg.get("provider", "anthropic")).strip().lower()
        has_anthropic = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
        has_gemini = bool((os.environ.get("GEMINI_API_KEY") or "").strip())
        if provider == "gemini" and (not has_gemini) and has_anthropic:
            provider = "anthropic"
        if provider == "gemini":
            self._impl: ReasoningBackend = GeminiReasoning(
                api_key=api_key,
                model_name=str(model_cfg.get("gemini_model", DEFAULT_GEMINI_MODEL)),
            )
        elif provider == "groq":
            self._impl = GroqReasoning(
                api_key=api_key,
                model_name=str(model_cfg.get("groq_model", DEFAULT_GROQ_MODEL)),
            )
        else:
            self._impl = ClaudeReasoning(
                api_key=api_key,
                model_name=str(model_cfg.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)),
            )

    def plan_tool_call(
        self,
        user_message: str,
        *,
        contract: RMICContract | None,
        condition: Condition,
        extra_system: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> PlannedToolCall:
        return self._impl.plan_tool_call(
            user_message,
            contract=contract,
            condition=condition,
            extra_system=extra_system,
            timeout_seconds=timeout_seconds,
        )


class ReasoningBackend(Protocol):
    def plan_tool_call(
        self,
        user_message: str,
        *,
        contract: RMICContract | None,
        condition: Condition,
        extra_system: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> PlannedToolCall: ...


class ClaudeReasoning:
    """Anthropic Claude backend through LiteLLM."""

    def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        self._api_key = (api_key or _api_key()).strip()
        self._model_name = _normalize_model_name(
            "anthropic",
            model_name or _model_name(),
        )
        self._provider = "anthropic"
        self._delay_seconds = 1.0

    def plan_tool_call(
        self,
        user_message: str,
        *,
        contract: RMICContract | None,
        condition: Condition,
        extra_system: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> PlannedToolCall:
        system = _system_prompt(contract, condition)
        if extra_system:
            system = f"{system}\n\n{extra_system}"
        return _plan_with_litellm(
            api_key=self._api_key,
            provider=self._provider,
            model_name=self._model_name,
            system=system,
            user_message=user_message,
            timeout_seconds=timeout_seconds,
            delay_seconds=self._delay_seconds,
        )


class GeminiReasoning:
    """Gemini backend through LiteLLM."""

    def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_GEMINI_MODEL) -> None:
        key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._api_key = key
        self._model_name = _normalize_model_name(
            "gemini",
            model_name or DEFAULT_GEMINI_MODEL,
        )
        self._provider = "gemini"
        self._delay_seconds = 4.0

    def plan_tool_call(
        self,
        user_message: str,
        *,
        contract: RMICContract | None,
        condition: Condition,
        extra_system: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> PlannedToolCall:
        system = _system_prompt(contract, condition)
        if extra_system:
            system = f"{system}\n\n{extra_system}"
        return _plan_with_litellm(
            api_key=self._api_key,
            provider=self._provider,
            model_name=self._model_name,
            system=system,
            user_message=user_message,
            timeout_seconds=timeout_seconds,
            delay_seconds=self._delay_seconds,
        )


class GroqReasoning:
    """Groq backend through LiteLLM."""

    def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_GROQ_MODEL) -> None:
        key = (api_key or os.environ.get("GROQ_API_KEY") or "").strip()
        if not key:
            raise ValueError("GROQ_API_KEY is not set")
        self._api_key = key
        self._model_name = _normalize_model_name(
            "groq",
            model_name or DEFAULT_GROQ_MODEL,
        )
        self._provider = "groq"
        self._delay_seconds = 1.1

    def plan_tool_call(
        self,
        user_message: str,
        *,
        contract: RMICContract | None,
        condition: Condition,
        extra_system: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> PlannedToolCall:
        system = _system_prompt(contract, condition)
        if extra_system:
            system = f"{system}\n\n{extra_system}"
        return _plan_with_litellm(
            api_key=self._api_key,
            provider=self._provider,
            model_name=self._model_name,
            system=system,
            user_message=user_message,
            timeout_seconds=timeout_seconds,
            delay_seconds=self._delay_seconds,
        )


def _plan_with_litellm(
    *,
    api_key: str,
    provider: str,
    model_name: str,
    system: str,
    user_message: str,
    timeout_seconds: float,
    delay_seconds: float,
) -> PlannedToolCall:
    max_attempts = 3
    backoff = 1.0
    last_text = ""
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            response = litellm.completion(
                model=model_name,
                api_key=api_key,
                temperature=0.2,
                max_tokens=1024,
                timeout=timeout_seconds,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{user_message}\n\n{_user_instructions_for_plan()}"},
                ],
            )
            last_text = str(response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"[MODEL] model={model_name} latency_ms={latency_ms} token_usage={usage}")

            plan, ok = ResponseValidator.validate_and_parse(last_text, provider)
            if ok:
                time.sleep(delay_seconds)
                return plan
            print(f"[VALIDATION_FAIL] attempt={attempt} model={model_name}")

            correction = litellm.completion(
                model=model_name,
                api_key=api_key,
                temperature=0.2,
                max_tokens=1024,
                timeout=timeout_seconds,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{user_message}\n\n{_user_instructions_for_plan()}"},
                    {"role": "assistant", "content": last_text},
                    {"role": "user", "content": ResponseValidator.CORRECTION_PROMPT},
                ],
            )
            corrected_text = str(correction.choices[0].message.content or "").strip()
            corrected_plan, corrected_ok = ResponseValidator.validate_and_parse(
                corrected_text, f"{provider}:correction"
            )
            if corrected_ok:
                time.sleep(delay_seconds)
                return corrected_plan
        except RateLimitError:
            if attempt >= max_attempts:
                break
            wait_seconds = min(30.0, backoff)
            time.sleep(wait_seconds)
            backoff *= 2.0
    return PlannedToolCall(
        tool_name="refused",
        arguments={},
        raw_text=last_text,
        data_categories_accessed=(),
    )
