"""Re-anchoring recovery protocol for warn-zone IDS."""

from __future__ import annotations

from core.contract_loader import RMICContract

__all__ = ["reanchoring_system_message", "reanchoring_user_nudge"]


def reanchoring_system_message(contract: RMICContract) -> str:
    """System-level re-anchor text derived only from sealed semantic_anchors."""
    lines = "\n".join(f"- {a}" for a in contract.semantic_anchors)
    return (
        "You are operating under a strict identity contract. Re-align with these anchors:\n"
        f"{lines}\n"
        "Produce only actions and outputs consistent with these anchors."
    )


def reanchoring_user_nudge(contract: RMICContract, reason: str) -> str:
    """User message appended on recovery retry after a warn-threshold event."""
    return (
        f"[RMIC recovery: {reason}]\n"
        "Re-evaluate your planned action against the identity anchors and respond with a revised plan."
    )
