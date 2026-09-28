"""Prompt selection knobs shared by applications and the comparison CLI."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import PromptBudgets, PromptLimits


@dataclass(frozen=True)
class PromptPolicy:
    total_tokens: int = 8000
    safety_margin_tokens: int = 500
    procedural_tokens: int = 1000
    episodic_event_tokens: int = 1000
    episodic_digest_tokens: int = 500
    semantic_tokens: int = 1000
    maximum_events: int = 6
    maximum_digests: int = 2
    maximum_semantic_documents: int = 3
    semantic_score_threshold: float = 0.30
    digest_score_threshold: float = 0.40
    episodic_event_max_chars: int | None = 4000
    episodic_digest_max_chars: int | None = 1800
    semantic_item_max_chars: int | None = 1800

    def budgets(self, *, include_semantic: bool = True) -> PromptBudgets:
        return PromptBudgets(
            total_tokens=self.total_tokens,
            procedural_tokens=self.procedural_tokens,
            episodic_event_tokens=self.episodic_event_tokens,
            episodic_digest_tokens=self.episodic_digest_tokens,
            semantic_tokens=self.semantic_tokens if include_semantic else 0,
            safety_margin_tokens=self.safety_margin_tokens,
        )

    def limits(self) -> PromptLimits:
        return PromptLimits(
            maximum_total_tokens=self.total_tokens,
            maximum_procedural_tokens=max(self.procedural_tokens, 1),
            maximum_episodic_event_tokens=max(self.episodic_event_tokens, 1),
            maximum_episodic_digest_tokens=max(self.episodic_digest_tokens, 1),
            maximum_semantic_tokens=max(self.semantic_tokens, 1),
            maximum_events=self.maximum_events,
            maximum_digests=self.maximum_digests,
            maximum_semantic_documents=self.maximum_semantic_documents,
            maximum_episodic_event_chars=self.episodic_event_max_chars,
            maximum_episodic_digest_chars=self.episodic_digest_max_chars,
            maximum_semantic_item_chars=self.semantic_item_max_chars,
        )
