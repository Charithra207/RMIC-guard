"""
Dual enforcement between reasoning output and tool execution.

Pass 1: hard rules (forbidden tools, parameter bounds, data scope).
Pass 2: IDS composite; block only when IDS >= block threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from core.audit_ledger import AuditEntry, AuditLedger, hash_text
from core.contract_loader import RMICContract
from core.ids_metric import compute_ids_components
from core.recovery_engine import reanchoring_system_message, reanchoring_user_nudge
from core.reasoning_layer import PlannedToolCall
from core.tool_layer import ToolRegistry, ToolResult

__all__ = ["EnforcementEngine", "EnforcementMode", "EnforcementOutcome"]

Decision = Literal["PASS", "BLOCK", "WARN", "NEEDS_RECOVERY", "PREEMPTIVE_WARN"]

EnforcementMode = Literal["full", "hard_rules_only", "ids_only"]


@dataclass(frozen=True)
class EnforcementOutcome:
    decision: Decision
    ids_score: float
    drift_velocity: float
    ids_components: dict[str, float] | None
    hard_rule_violation: str | None
    recovery_system_message: str | None
    recovery_user_message: str | None
    tool_result: ToolResult | None


def _numeric_for_type(val: Any, value_type: str) -> float:
    if value_type == "int":
        return float(int(val))
    return float(val)


def _check_parameter_constraints(
    contract: RMICContract,
    arguments: dict[str, Any],
) -> str | None:
    by_name = contract.constraints_by_name()
    for name, spec in by_name.items():
        if name not in arguments:
            continue
        try:
            v = _numeric_for_type(arguments[name], spec.value_type)
        except (TypeError, ValueError):
            return f"parameter_constraint:{name}:not_numeric"
        if spec.max is not None and v > float(spec.max):
            return f"parameter_constraint:{name}:above_max"
        if spec.min is not None and v < float(spec.min):
            return f"parameter_constraint:{name}:below_min"
    return None


def _check_data_scope(contract: RMICContract, plan: PlannedToolCall) -> str | None:
    prohibited = set(contract.data_scope.prohibited)
    for cat in plan.data_categories_accessed:
        if cat in prohibited:
            return f"data_scope:prohibited:{cat}"
    return None


def _check_hard_rules(contract: RMICContract, plan: PlannedToolCall) -> str | None:
    if plan.tool_name in contract.forbidden_actions:
        return "forbidden_tool"
    if contract.allowed_actions and plan.tool_name not in contract.allowed_actions:
        return "tool_not_allowed"
    pr = _check_parameter_constraints(contract, plan.arguments)
    if pr:
        return pr
    return _check_data_scope(contract, plan)


def _ids_on_plan(
    contract: RMICContract,
    plan: PlannedToolCall,
    recent_ids: list[float],
    tool_call_history: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute IDS score and component breakdown for a planned tool call.

    Args:
        contract: The sealed RMIC contract.
        plan: Planned tool call with semantic context.
        recent_ids: History of prior IDS scores (for trajectory curvature).
        tool_call_history: Sequence of tool names called in this session.

    Returns:
        Tuple of (base_ids_score, components_dict). The components dictionary
        contains all independent drift metrics used for dashboard visualization.

    Raises:
        ValueError: If contract.anchor_embedding is empty (contract not sealed).
    """
    if not contract.anchor_embedding:
        raise ValueError("contract missing anchor_embedding — run seal_contract_file first")
    text_for_ids = f"{plan.tool_name} {plan.arguments} {plan.raw_text}"
    allowed_topics = list(contract.semantic_anchors)
    forbidden_topics: list[str] = []
    if contract.data_scope.prohibited:
        forbidden_topics.extend(contract.data_scope.prohibited)
    if contract.forbidden_actions:
        forbidden_topics.extend(contract.forbidden_actions)
    components = compute_ids_components(
        text_for_ids,
        contract.anchor_embedding,
        allowed_topics=allowed_topics,
        forbidden_topics=forbidden_topics,
        recent_ids=recent_ids,
        tool_call_history=list(tool_call_history or []) + [plan.tool_name],
        allowed_actions=tuple(contract.allowed_actions),
    )
    return components["base_ids"], components


def _velocity(recent_ids: list[float], new_ids: float) -> float:
    if not recent_ids:
        return 0.0
    return abs(new_ids - recent_ids[-1])


class EnforcementEngine:
    def __init__(
        self,
        contract: RMICContract,
        tools: ToolRegistry,
        ledger: AuditLedger | None = None,
        log_async: Callable[[AuditEntry], None] | None = None,
        ids_warn_threshold_override: float | None = None,
        ids_block_threshold_override: float | None = None,
    ) -> None:
        self.contract = contract
        self.tools = tools
        self.ledger = ledger
        self.log_async = log_async
        self.ids_warn_threshold = (
            ids_warn_threshold_override
            if ids_warn_threshold_override is not None
            else contract.ids_warn_threshold
        )
        self.ids_block_threshold = (
            ids_block_threshold_override
            if ids_block_threshold_override is not None
            else contract.ids_block_threshold
        )

    def _log(self, entry: AuditEntry, *, sync: bool) -> None:
        if self.ledger is None:
            return
        if sync:
            self.ledger.append(entry)
        elif self.log_async:
            self.log_async(entry)
        else:
            self.ledger.append(entry)

    def evaluate_and_maybe_execute(
        self,
        plan: PlannedToolCall,
        *,
        recent_ids: list[float],
        tool_call_history: list[str] | None = None,
        drift_type: str | None = None,
        execute_tool: bool = True,
        enforcement_mode: EnforcementMode = "full",
    ) -> EnforcementOutcome:
        """
        Run dual enforcement. Executes the tool only when decision is PASS and execute_tool is True.
        Below warn threshold: append audit with sync=False when log_async is set (caller may flush).

        enforcement_mode:
          full — hard rules then IDS (default).
          hard_rules_only — Pass 1 only; if hard rules pass, PASS with ids_score=0 (no IDS).
          ids_only — Pass 2 only; skip hard rules, block only on IDS thresholds.
        """
        c = self.contract
        if enforcement_mode == "ids_only":
            hard = None
        else:
            hard = _check_hard_rules(c, plan)
        if hard:
            # Compute IDS even on hard-rule block so the dashboard always has
            # real embedding scores for every Condition C row.
            ids_score_measured = 0.0
            ids_components_measured: dict[str, float] | None = None
            if c.anchor_embedding and enforcement_mode not in ("hard_rules_only",):
                try:
                    ids_score_measured, ids_components_measured = _ids_on_plan(
                        c,
                        plan,
                        recent_ids,
                        tool_call_history=tool_call_history,
                    )
                except Exception:
                    ids_score_measured = 0.0
                    ids_components_measured = None
            self._log(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent_id=c.agent_id,
                    input_hash=hash_text(plan.raw_text),
                    ids_score=ids_score_measured,
                    drift_type=drift_type,
                    drift_velocity=None,
                    decision="BLOCK",
                    contract_hash=c.contract_hash,
                    recovery_attempted=False,
                ),
                sync=True,
            )
            return EnforcementOutcome(
                decision="BLOCK",
                ids_score=ids_score_measured,
                drift_velocity=0.0,
                ids_components=ids_components_measured,
                hard_rule_violation=hard,
                recovery_system_message=None,
                recovery_user_message=None,
                tool_result=None,
            )

        if enforcement_mode == "hard_rules_only":
            pass_sync = self.log_async is None
            self._log(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent_id=c.agent_id,
                    input_hash=hash_text(plan.raw_text),
                    ids_score=0.0,
                    drift_type=drift_type,
                    drift_velocity=None,
                    decision="PASS",
                    contract_hash=c.contract_hash,
                    recovery_attempted=False,
                ),
                sync=pass_sync,
            )
            tool_result: ToolResult | None = None
            if execute_tool:
                tool_result = self.tools.execute(plan.tool_name, **plan.arguments)
            return EnforcementOutcome(
                decision="PASS",
                ids_score=0.0,
                drift_velocity=0.0,
                ids_components=None,
                hard_rule_violation=None,
                recovery_system_message=None,
                recovery_user_message=None,
                tool_result=tool_result,
            )

        ids_score, ids_components = _ids_on_plan(
            c,
            plan,
            recent_ids,
            tool_call_history=tool_call_history,
        )
        vel = _velocity(recent_ids, ids_score)

        def audit(decision: str, recovery_attempted: bool, *, sync_log: bool) -> None:
            self._log(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent_id=c.agent_id,
                    input_hash=hash_text(plan.raw_text),
                    ids_score=ids_score,
                    drift_type=drift_type,
                    drift_velocity=vel,
                    decision=decision,
                    contract_hash=c.contract_hash,
                    recovery_attempted=recovery_attempted,
                ),
                sync=sync_log,
            )

        if ids_score >= self.ids_block_threshold:
            audit("BLOCK", False, sync_log=True)
            return EnforcementOutcome(
                decision="BLOCK",
                ids_score=ids_score,
                drift_velocity=vel,
                ids_components=ids_components,
                hard_rule_violation=None,
                recovery_system_message=None,
                recovery_user_message=None,
                tool_result=None,
            )

        preemptive = vel > c.drift_velocity_threshold and ids_score < self.ids_warn_threshold

        if ids_score >= self.ids_warn_threshold:
            audit("WARN", True, sync_log=True)
            return EnforcementOutcome(
                decision="NEEDS_RECOVERY",
                ids_score=ids_score,
                drift_velocity=vel,
                ids_components=ids_components,
                hard_rule_violation=None,
                recovery_system_message=reanchoring_system_message(c),
                recovery_user_message=reanchoring_user_nudge(
                    c,
                    reason=f"IDS {ids_score:.3f} in warn zone (>={self.ids_warn_threshold})",
                ),
                tool_result=None,
            )

        pass_sync = self.log_async is None
        final_decision: str = "PREEMPTIVE_WARN" if preemptive else "PASS"
        audit(final_decision, False, sync_log=pass_sync)

        tool_result: ToolResult | None = None
        if execute_tool:
            tool_result = self.tools.execute(plan.tool_name, **plan.arguments)

        decision: Decision
        if final_decision == "PREEMPTIVE_WARN":
            decision = "PREEMPTIVE_WARN"
        else:
            decision = "PASS"
        return EnforcementOutcome(
            decision=decision,
            ids_score=ids_score,
            drift_velocity=vel,
            ids_components=ids_components,
            hard_rule_violation=None,
            recovery_system_message=None,
            recovery_user_message=None,
            tool_result=tool_result,
        )
