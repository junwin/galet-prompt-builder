from __future__ import annotations

from typing import Sequence

from .candidates import PromptCandidate
from .contracts import PromptMessage


_SOURCE_ORDER = {
    "procedural": 0,
    "semantic": 1,
    "episodic_digest": 2,
    "episodic_event": 3,
}


def render_memory_messages(
    candidates: Sequence[PromptCandidate],
) -> list[PromptMessage]:
    ordered = sorted(
        candidates,
        key=lambda item: (_SOURCE_ORDER[item.source], item.order),
    )
    return [
        PromptMessage(
            role=item.role,
            content=item.content,
            source=item.source,
            source_id=item.candidate_id,
        )
        for item in ordered
    ]


def render_prompt_text(messages: Sequence[PromptMessage]) -> str:
    return "\n\n".join(
        f"[{message.role}]\n{message.content}" for message in messages
    )


__all__ = ["render_memory_messages", "render_prompt_text"]
