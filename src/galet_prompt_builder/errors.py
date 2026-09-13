class PromptBuilderError(RuntimeError):
    """Base class for prompt compilation failures."""


class PromptConfigurationError(PromptBuilderError, ValueError):
    """Requested budgets or limits are invalid."""


class PromptBudgetExceededError(PromptBuilderError):
    """Mandatory prompt content cannot fit the total budget."""


class MemoryRetrievalError(PromptBuilderError):
    """A configured memory source could not be recalled."""


class RelevanceAssessmentError(PromptBuilderError):
    """Candidate relevance could not be assessed safely."""


__all__ = [
    "MemoryRetrievalError",
    "PromptBudgetExceededError",
    "PromptBuilderError",
    "PromptConfigurationError",
    "RelevanceAssessmentError",
]
