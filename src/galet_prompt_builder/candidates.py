from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import PromptSource


@dataclass(frozen=True)
class PromptCandidate:
    candidate_id: str
    source: PromptSource
    role: str
    content: str
    relevance: float = 0.0
    required: bool = False
    order: int = 0
    truncated: bool = False

    def with_relevance(self, relevance: float) -> "PromptCandidate":
        return replace(self, relevance=max(0.0, min(1.0, relevance)))


__all__ = ["PromptCandidate"]
