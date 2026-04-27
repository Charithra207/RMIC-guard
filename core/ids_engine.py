"""Clean IDS engine wrapper for testable scoring calls."""

from __future__ import annotations

from typing import Sequence

from core.ids_metric import compute_ids_components


class IDSEngine:
    def score(
        self,
        text_for_ids: str,
        anchor_embedding: Sequence[float],
        *,
        allowed_topics: Sequence[str],
        forbidden_topics: Sequence[str],
        recent_ids: Sequence[float],
        tool_call_history: list[str],
        allowed_actions: tuple[str, ...],
    ) -> dict[str, float]:
        return compute_ids_components(
            text_for_ids,
            anchor_embedding,
            allowed_topics=allowed_topics,
            forbidden_topics=forbidden_topics,
            recent_ids=recent_ids,
            tool_call_history=tool_call_history,
            allowed_actions=allowed_actions,
        )

