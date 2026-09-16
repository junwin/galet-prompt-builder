from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import PromptMessage, PromptSource


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    source: PromptSource
    relevance: float
    original_tokens: int
    selected: bool
    final_tokens: int = 0
    truncated: bool = False
    required: bool = False
    drop_reason: str = ""


@dataclass(frozen=True)
class SectionMetrics:
    budget_tokens: int
    used_tokens: int = 0
    retrieved_items: int = 0
    selected_items: int = 0
    dropped_items: int = 0
    truncated_items: int = 0


@dataclass(frozen=True)
class PromptMetrics:
    total_limit_tokens: int
    total_used_tokens: int
    safety_margin_tokens: int
    system: SectionMetrics
    procedural: SectionMetrics
    episodic_events: SectionMetrics
    episodic_digests: SectionMetrics
    semantic: SectionMetrics
    current_input: SectionMetrics
    warnings: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[CandidateMetrics, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompiledPrompt:
    messages: tuple[PromptMessage, ...]
    metrics: PromptMetrics

    @property
    def provider_messages(self) -> list[dict[str, str]]:
        return [message.as_provider_message() for message in self.messages]

    @property
    def text(self) -> str:
        from .rendering import render_prompt_text

        return render_prompt_text(self.messages)


__all__ = [
    "CandidateMetrics",
    "CompiledPrompt",
    "PromptMetrics",
    "SectionMetrics",
]
