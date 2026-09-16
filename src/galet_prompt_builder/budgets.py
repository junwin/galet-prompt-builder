from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .contracts import PromptBudgets, PromptLimits
from .errors import PromptConfigurationError


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...


class ApproximateTokenCounter:
    """Deterministic fallback using approximately four characters per token."""

    def count(self, text: str) -> int:
        return 0 if not text else max(1, math.ceil(len(text) / 4))


def validate_budgets(budgets: PromptBudgets, limits: PromptLimits) -> None:
    values = {
        "total_tokens": budgets.total_tokens,
        "procedural_tokens": budgets.procedural_tokens,
        "episodic_event_tokens": budgets.episodic_event_tokens,
        "episodic_digest_tokens": budgets.episodic_digest_tokens,
        "semantic_tokens": budgets.semantic_tokens,
        "safety_margin_tokens": budgets.safety_margin_tokens,
    }
    negative = [name for name, value in values.items() if value < 0]
    if negative:
        raise PromptConfigurationError(
            "token budgets must not be negative: " + ", ".join(negative)
        )
    maxima = {
        "total_tokens": limits.maximum_total_tokens,
        "procedural_tokens": limits.maximum_procedural_tokens,
        "episodic_event_tokens": limits.maximum_episodic_event_tokens,
        "episodic_digest_tokens": limits.maximum_episodic_digest_tokens,
        "semantic_tokens": limits.maximum_semantic_tokens,
    }
    exceeded = [
        name for name, maximum in maxima.items() if values[name] > maximum
    ]
    if exceeded:
        raise PromptConfigurationError(
            "requested budgets exceed configured maxima: "
            + ", ".join(exceeded)
        )
    if budgets.safety_margin_tokens >= budgets.total_tokens:
        raise PromptConfigurationError(
            "safety margin must be smaller than the total token budget"
        )
    limit_values = {
        "maximum_total_tokens": limits.maximum_total_tokens,
        "maximum_procedural_tokens": limits.maximum_procedural_tokens,
        "maximum_episodic_event_tokens": limits.maximum_episodic_event_tokens,
        "maximum_episodic_digest_tokens": limits.maximum_episodic_digest_tokens,
        "maximum_semantic_tokens": limits.maximum_semantic_tokens,
        "maximum_events": limits.maximum_events,
        "maximum_digests": limits.maximum_digests,
        "maximum_semantic_documents": limits.maximum_semantic_documents,
        "maximum_item_chars": limits.maximum_item_chars,
    }
    optional_limit_values = {
        "maximum_episodic_event_chars": limits.maximum_episodic_event_chars,
        "maximum_episodic_digest_chars": limits.maximum_episodic_digest_chars,
        "maximum_semantic_item_chars": limits.maximum_semantic_item_chars,
    }
    limit_values.update(
        {
            name: value
            for name, value in optional_limit_values.items()
            if value is not None
        }
    )
    non_positive = [name for name, value in limit_values.items() if value <= 0]
    if non_positive:
        raise PromptConfigurationError(
            "configured limits must be greater than zero: "
            + ", ".join(non_positive)
        )
    if limits.message_overhead_tokens < 0:
        raise PromptConfigurationError(
            "message_overhead_tokens must not be negative"
        )


def truncate_to_tokens(
    text: str,
    maximum_tokens: int,
    counter: TokenCounter,
    *,
    suffix: str = "…",
) -> str:
    if maximum_tokens <= 0:
        return ""
    if counter.count(text) <= maximum_tokens:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle].rstrip() + suffix
        if counter.count(candidate) <= maximum_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


__all__ = [
    "ApproximateTokenCounter",
    "TokenCounter",
    "truncate_to_tokens",
    "validate_budgets",
]
