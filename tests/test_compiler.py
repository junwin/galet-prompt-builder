from types import SimpleNamespace

import pytest

from galet_memory import (
    EpisodicDigest,
    EpisodicEvent,
    EpisodicMemoryResult,
    ProceduralMemoryResult,
    ProceduralSkill,
    SemanticDocument,
    SemanticMemoryResult,
)
from galet_prompt_builder import (
    MemoryRetrievalError,
    PromptBudgetExceededError,
    PromptBudgets,
    PromptCompiler,
    PromptConfigurationError,
    PromptLimits,
    PromptRequest,
)


class WordCounter:
    def count(self, text):
        return len(text.split())


class CharacterCounter:
    def count(self, text):
        return len(text)


class FakeProceduralMemory:
    def __init__(self):
        self.request = None

    def recall(self, request):
        self.request = request
        return ProceduralMemoryResult(
            context_id="project",
            account_name=request.account_name,
            resolved_text="Use the project conventions.",
            skills=[ProceduralSkill("testing", "Run focused tests.")],
        )


class FakeEpisodicMemory:
    def __init__(self, result=None, error=None):
        self.result = result or EpisodicMemoryResult(
            session_id="session",
            events=[
                EpisodicEvent("user", "Earlier question", event_id="e1"),
                EpisodicEvent(
                    "assistant", "Earlier answer", event_id="e2"
                ),
            ],
            digests=[EpisodicDigest("old-session", "Older decision", 0.8)],
        )
        self.error = error
        self.request = None

    def recall(self, request):
        self.request = request
        if self.error:
            raise self.error
        return self.result

    def save_overflow_digest(self, **kwargs):
        raise AssertionError("prompt compilation must be read-only")


class FakeSemanticMemory:
    def __init__(self, documents=None):
        self.documents = documents or [
            SemanticDocument(
                "doc-1", "Storage design", "SQLite remains compatible.", score=0.9
            )
        ]
        self.request = None

    def recall(self, request):
        self.request = request
        return SemanticMemoryResult(documents=self.documents)


def _limits(**overrides):
    values = dict(
        maximum_total_tokens=200,
        maximum_procedural_tokens=50,
        maximum_episodic_event_tokens=50,
        maximum_episodic_digest_tokens=50,
        maximum_semantic_tokens=50,
        maximum_events=10,
        maximum_digests=5,
        maximum_semantic_documents=5,
        message_overhead_tokens=0,
    )
    values.update(overrides)
    return PromptLimits(**values)


def _budgets(**overrides):
    values = dict(
        total_tokens=150,
        procedural_tokens=30,
        episodic_event_tokens=30,
        episodic_digest_tokens=30,
        semantic_tokens=30,
        safety_margin_tokens=10,
    )
    values.update(overrides)
    return PromptBudgets(**values)


def test_compiles_all_memory_types_into_structured_prompt():
    procedural = FakeProceduralMemory()
    episodic = FakeEpisodicMemory()
    semantic = FakeSemanticMemory()
    compiler = PromptCompiler(
        procedural_memory=procedural,
        episodic_memory=episodic,
        semantic_memory=semantic,
        token_counter=WordCounter(),
    )

    compiled = compiler.compile(
        PromptRequest(
            account_name="acct",
            current_input="What did we decide?",
            system_instructions=("Be careful.",),
            conversation_id="session",
            context_name="project",
        ),
        _budgets(),
        _limits(),
    )

    sources = [message.source for message in compiled.messages]
    assert sources[0] == "system"
    assert sources[-1] == "current_input"
    assert {
        "procedural",
        "episodic_event",
        "episodic_digest",
        "semantic",
    } <= set(sources)
    assert compiled.provider_messages[0] == {
        "role": "system",
        "content": "Be careful.",
    }
    assert "source" not in compiled.provider_messages[0]
    assert "[user]" in compiled.text
    assert procedural.request.create_if_missing is False
    assert episodic.request.agent_name == ""
    assert semantic.request.query == "What did we decide?"
    assert compiled.metrics.total_used_tokens <= 140


def test_mandatory_content_that_cannot_fit_fails_clearly():
    compiler = PromptCompiler(token_counter=WordCounter())
    with pytest.raises(PromptBudgetExceededError, match="mandatory"):
        compiler.compile(
            PromptRequest(
                account_name="acct",
                current_input="one two three four",
                system_instructions=("five six seven",),
            ),
            _budgets(total_tokens=7, safety_margin_tokens=1),
            _limits(),
        )


def test_section_budget_truncates_without_exceeding_limit():
    semantic = FakeSemanticMemory(
        [SemanticDocument("doc", "Title", "x" * 100, score=1.0)]
    )
    compiled = PromptCompiler(
        semantic_memory=semantic,
        token_counter=CharacterCounter(),
    ).compile(
        PromptRequest(account_name="acct", current_input="q"),
        _budgets(
            total_tokens=100,
            procedural_tokens=0,
            episodic_event_tokens=0,
            episodic_digest_tokens=0,
            semantic_tokens=20,
            safety_margin_tokens=0,
        ),
        _limits(),
    )

    metric = compiled.metrics.semantic
    assert metric.used_tokens <= 20
    assert metric.selected_items == 1
    assert metric.truncated_items == 1
    assert compiled.metrics.total_used_tokens <= 100


def test_character_limit_includes_the_truncation_marker():
    semantic = FakeSemanticMemory(
        [SemanticDocument("doc", "Title", "x" * 100, score=1.0)]
    )
    compiled = PromptCompiler(
        semantic_memory=semantic,
        token_counter=CharacterCounter(),
    ).compile(
        PromptRequest(account_name="acct", current_input="q"),
        _budgets(
            total_tokens=100,
            procedural_tokens=0,
            episodic_event_tokens=0,
            episodic_digest_tokens=0,
            semantic_tokens=50,
            safety_margin_tokens=0,
        ),
        _limits(maximum_item_chars=20),
    )
    message = next(
        item for item in compiled.messages if item.source == "semantic"
    )
    assert len(message.content) <= 20
    assert message.content.endswith("…")


def test_higher_scored_semantic_document_is_selected_first():
    semantic = FakeSemanticMemory(
        [
            SemanticDocument("low", "Low", "low text", score=0.1),
            SemanticDocument("high", "High", "high text", score=0.9),
        ]
    )
    compiled = PromptCompiler(
        semantic_memory=semantic,
        token_counter=WordCounter(),
    ).compile(
        PromptRequest(account_name="acct", current_input="unrelated query"),
        _budgets(
            procedural_tokens=0,
            episodic_event_tokens=0,
            episodic_digest_tokens=0,
            semantic_tokens=5,
        ),
        _limits(),
    )
    semantic_messages = [
        message for message in compiled.messages if message.source == "semantic"
    ]
    assert len(semantic_messages) == 1
    assert "High" in semantic_messages[0].content
    assert compiled.metrics.semantic.dropped_items == 1


def test_duplicate_active_and_recalled_digest_is_inserted_once():
    result = EpisodicMemoryResult(
        session_id="session",
        events=[
            EpisodicEvent(
                "system",
                "same digest",
                kind="session_digest",
                event_id="digest-event",
                metadata={"visibility_boundary": True},
            )
        ],
        digests=[EpisodicDigest("session", "same digest", 1.0)],
    )
    compiled = PromptCompiler(
        episodic_memory=FakeEpisodicMemory(result),
        token_counter=WordCounter(),
    ).compile(
        PromptRequest(account_name="acct", current_input="question"),
        _budgets(),
        _limits(),
    )
    assert [message.content for message in compiled.messages].count(
        "same digest"
    ) == 1
    assert compiled.metrics.episodic_digests.retrieved_items == 0


def test_memory_failure_is_not_silently_hidden():
    compiler = PromptCompiler(
        episodic_memory=FakeEpisodicMemory(error=RuntimeError("offline"))
    )
    with pytest.raises(MemoryRetrievalError, match="episodic"):
        compiler.compile(
            PromptRequest(account_name="acct", current_input="question"),
            _budgets(),
            _limits(),
        )


def test_metrics_do_not_copy_memory_content():
    compiled = PromptCompiler(
        episodic_memory=FakeEpisodicMemory(), token_counter=WordCounter()
    ).compile(
        PromptRequest(account_name="acct", current_input="question"),
        _budgets(),
        _limits(),
    )
    assert "Earlier question" not in repr(compiled.metrics)


def test_episodic_policy_filters_event_kinds_and_structured_payloads():
    result = EpisodicMemoryResult(
        session_id="session",
        events=[
            EpisodicEvent(
                "user", "Keep this", kind="user_message", event_id="e1"
            ),
            EpisodicEvent(
                "system",
                {"total": 6000},
                kind="prompt_report",
                event_id="e2",
            ),
            EpisodicEvent(
                "assistant",
                {"tool_name": "delegate_task"},
                kind="assistant_message",
                event_id="e3",
            ),
        ],
    )
    episodic = FakeEpisodicMemory(result)

    compiled = PromptCompiler(
        episodic_memory=episodic,
        token_counter=WordCounter(),
    ).compile(
        PromptRequest(
            account_name="acct",
            current_input="question",
            episodic_event_kinds=("user_message", "assistant_message"),
            include_structured_episodic_events=False,
        ),
        _budgets(),
        _limits(maximum_events=6),
    )

    episodic_messages = [
        item for item in compiled.messages if item.source == "episodic_event"
    ]
    assert [item.content for item in episodic_messages] == ["Keep this"]
    assert episodic.request.max_events == 6
    assert episodic.request.event_kinds == [
        "user_message",
        "assistant_message",
    ]


def test_digest_threshold_drops_weak_archived_digest():
    result = EpisodicMemoryResult(
        session_id="session",
        digests=[
            EpisodicDigest("weak", "Weak digest", 0.29),
            EpisodicDigest("strong", "Strong digest", 0.75),
        ],
    )

    compiled = PromptCompiler(
        episodic_memory=FakeEpisodicMemory(result),
        token_counter=WordCounter(),
    ).compile(
        PromptRequest(
            account_name="acct",
            current_input="question",
            episodic_digest_score_threshold=0.4,
        ),
        _budgets(),
        _limits(),
    )

    digest_messages = [
        item for item in compiled.messages if item.source == "episodic_digest"
    ]
    assert len(digest_messages) == 1
    assert "Strong digest" in digest_messages[0].content


def test_semantic_policy_is_forwarded_and_item_size_is_bounded():
    semantic = FakeSemanticMemory(
        [
            SemanticDocument(
                "doc",
                "Title",
                "x" * 100,
                score=0.9,
            )
        ]
    )

    compiled = PromptCompiler(
        semantic_memory=semantic,
        token_counter=CharacterCounter(),
    ).compile(
        PromptRequest(
            account_name="acct",
            current_input="question",
            semantic_score_threshold=0.35,
        ),
        _budgets(),
        _limits(maximum_semantic_item_chars=40),
    )

    semantic_message = next(
        item for item in compiled.messages if item.source == "semantic"
    )
    assert semantic.request.score_threshold == 0.35
    assert semantic.request.max_chars == 40
    assert len(semantic_message.content) <= 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_score_threshold", -0.1),
        ("semantic_score_threshold", 1.1),
        ("episodic_digest_score_threshold", -0.1),
        ("episodic_digest_score_threshold", 1.1),
    ],
)
def test_relevance_thresholds_must_be_probabilities(field, value):
    values = {
        "account_name": "acct",
        "current_input": "question",
        field: value,
    }
    with pytest.raises(PromptConfigurationError, match=field):
        PromptCompiler().compile(
            PromptRequest(**values),
            _budgets(),
            _limits(),
        )
