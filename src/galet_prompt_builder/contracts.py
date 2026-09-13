from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


PromptSource = Literal[
    "system",
    "procedural",
    "episodic_event",
    "episodic_digest",
    "semantic",
    "current_input",
]


@dataclass(frozen=True)
class PromptRequest:
    account_name: str
    current_input: str
    system_instructions: Sequence[str] = field(default_factory=tuple)
    conversation_id: str = ""
    context_name: str = ""
    semantic_query: str = ""
    semantic_namespaces: Sequence[str] = field(
        default_factory=lambda: ("external",)
    )
    include_procedural: bool = True
    include_episodic: bool = True
    include_semantic: bool = True
    include_digests: bool = True


@dataclass(frozen=True)
class PromptBudgets:
    total_tokens: int
    procedural_tokens: int
    episodic_event_tokens: int
    episodic_digest_tokens: int
    semantic_tokens: int
    safety_margin_tokens: int = 0


@dataclass(frozen=True)
class PromptLimits:
    maximum_total_tokens: int
    maximum_procedural_tokens: int
    maximum_episodic_event_tokens: int
    maximum_episodic_digest_tokens: int
    maximum_semantic_tokens: int
    maximum_events: int = 20
    maximum_digests: int = 5
    maximum_semantic_documents: int = 5
    maximum_item_chars: int = 12000
    message_overhead_tokens: int = 4


@dataclass(frozen=True)
class PromptMessage:
    role: str
    content: str
    source: PromptSource
    source_id: str = ""

    def as_provider_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


__all__ = [
    "PromptBudgets",
    "PromptLimits",
    "PromptMessage",
    "PromptRequest",
    "PromptSource",
]
