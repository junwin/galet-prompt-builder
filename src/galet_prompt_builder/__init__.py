"""Provider-neutral, memory-aware prompt compilation."""

from .budgets import ApproximateTokenCounter, TokenCounter
from .candidates import PromptCandidate
from .compiler import PromptCompiler
from .contracts import (
    PromptBudgets,
    PromptLimits,
    PromptMessage,
    PromptRequest,
    PromptSource,
)
from .errors import (
    MemoryRetrievalError,
    PromptBudgetExceededError,
    PromptBuilderError,
    PromptConfigurationError,
    RelevanceAssessmentError,
)
from .metrics import CompiledPrompt, PromptMetrics, SectionMetrics
from .relevance import (
    DeterministicRelevanceAssessor,
    GaletRelevanceAssessor,
    RelevanceAssessor,
)
from .rendering import render_prompt_text

__version__ = "0.1.0.dev0"

__all__ = [
    "ApproximateTokenCounter",
    "CompiledPrompt",
    "DeterministicRelevanceAssessor",
    "GaletRelevanceAssessor",
    "MemoryRetrievalError",
    "PromptBudgetExceededError",
    "PromptBudgets",
    "PromptBuilderError",
    "PromptCandidate",
    "PromptCompiler",
    "PromptConfigurationError",
    "PromptLimits",
    "PromptMessage",
    "PromptMetrics",
    "PromptRequest",
    "PromptSource",
    "RelevanceAssessmentError",
    "RelevanceAssessor",
    "SectionMetrics",
    "TokenCounter",
    "render_prompt_text",
    "__version__",
]
