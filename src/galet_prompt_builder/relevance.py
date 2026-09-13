from __future__ import annotations

import json
import re
from typing import Protocol, Sequence, runtime_checkable

from galet.interface import LLMApi

from .candidates import PromptCandidate
from .errors import RelevanceAssessmentError


@runtime_checkable
class RelevanceAssessor(Protocol):
    def assess(
        self,
        query: str,
        candidates: Sequence[PromptCandidate],
    ) -> Sequence[PromptCandidate]:
        ...


class DeterministicRelevanceAssessor:
    """Rank candidates using existing scores, recency, and word overlap."""

    _BASE_SCORES = {
        "procedural": 0.65,
        "episodic_event": 0.55,
        "episodic_digest": 0.45,
        "semantic": 0.50,
    }

    def assess(
        self,
        query: str,
        candidates: Sequence[PromptCandidate],
    ) -> Sequence[PromptCandidate]:
        query_words = self._words(query)
        largest_order = max((item.order for item in candidates), default=0)
        assessed: list[PromptCandidate] = []
        for candidate in candidates:
            score = candidate.relevance or self._BASE_SCORES.get(
                candidate.source, 0.0
            )
            content_words = self._words(candidate.content)
            if query_words and content_words:
                overlap = len(query_words & content_words) / len(query_words)
                score = max(score, min(1.0, 0.35 + 0.65 * overlap))
            if candidate.source == "episodic_event" and largest_order:
                recency = candidate.order / largest_order
                score = max(score, 0.45 + 0.35 * recency)
            if candidate.required:
                score = 1.0
            assessed.append(candidate.with_relevance(score))
        return assessed

    @staticmethod
    def _words(text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z0-9]+", text.casefold())
            if len(word) > 2
        }


class GaletRelevanceAssessor:
    """Optional LLM assessor using only Galet's provider-neutral API."""

    def __init__(self, llm_api: LLMApi, *, model: str) -> None:
        self.llm_api = llm_api
        self.model = model

    def assess(
        self,
        query: str,
        candidates: Sequence[PromptCandidate],
    ) -> Sequence[PromptCandidate]:
        if not candidates:
            return []
        payload = [
            {
                "id": item.candidate_id,
                "source": item.source,
                "text": item.content[:1000],
            }
            for item in candidates
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "Score each memory candidate for relevance to the query "
                    "from 0.0 to 1.0. Return only a JSON object mapping each "
                    "candidate id to its numeric score."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"query": query, "candidates": payload},
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = self.llm_api.create_response(
                model=self.model,
                input=messages,
            )
            scores = self._parse_scores(response.output_text)
        except Exception as exc:
            raise RelevanceAssessmentError(
                "Galet relevance assessment failed"
            ) from exc
        return [
            item.with_relevance(float(scores.get(item.candidate_id, 0.0)))
            if not item.required
            else item.with_relevance(1.0)
            for item in candidates
        ]

    @staticmethod
    def _parse_scores(text: str) -> dict[str, float]:
        value = text.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value)
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("relevance response must be a JSON object")
        scores: dict[str, float] = {}
        for key, score in parsed.items():
            numeric = float(score)
            scores[str(key)] = max(0.0, min(1.0, numeric))
        return scores


__all__ = [
    "DeterministicRelevanceAssessor",
    "GaletRelevanceAssessor",
    "RelevanceAssessor",
]
