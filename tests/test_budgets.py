import pytest

from galet_prompt_builder import (
    ApproximateTokenCounter,
    PromptBudgets,
    PromptConfigurationError,
    PromptLimits,
)
from galet_prompt_builder.budgets import truncate_to_tokens, validate_budgets


def _limits():
    return PromptLimits(
        maximum_total_tokens=100,
        maximum_procedural_tokens=20,
        maximum_episodic_event_tokens=20,
        maximum_episodic_digest_tokens=20,
        maximum_semantic_tokens=20,
    )


def test_requested_budget_cannot_exceed_configured_maximum():
    with pytest.raises(PromptConfigurationError, match="semantic_tokens"):
        validate_budgets(
            PromptBudgets(100, 10, 10, 10, 21),
            _limits(),
        )


def test_negative_budget_is_rejected():
    with pytest.raises(PromptConfigurationError, match="must not be negative"):
        validate_budgets(
            PromptBudgets(100, -1, 10, 10, 10),
            _limits(),
        )


def test_safety_margin_must_leave_usable_space():
    with pytest.raises(PromptConfigurationError, match="safety margin"):
        validate_budgets(
            PromptBudgets(100, 10, 10, 10, 10, safety_margin_tokens=100),
            _limits(),
        )


def test_approximate_counter_and_truncation_are_deterministic():
    counter = ApproximateTokenCounter()
    assert counter.count("12345") == 2
    shortened = truncate_to_tokens("abcdefghijk", 2, counter)
    assert shortened.endswith("…")
    assert counter.count(shortened) <= 2
