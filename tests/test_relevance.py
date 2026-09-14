import pytest

from galet_prompt_builder import (
    DeterministicRelevanceAssessor,
    GaletRelevanceAssessor,
    PromptCandidate,
    RelevanceAssessmentError,
)


def _candidate(identifier, text, *, required=False):
    return PromptCandidate(
        identifier,
        "semantic",
        "system",
        text,
        required=required,
    )


def test_deterministic_assessor_uses_overlap_and_required_priority():
    assessor = DeterministicRelevanceAssessor()
    results = assessor.assess(
        "sqlite storage",
        [
            _candidate("matching", "sqlite storage design"),
            _candidate("other", "photography notes"),
            _candidate("required", "unrelated", required=True),
        ],
    )
    scores = {item.candidate_id: item.relevance for item in results}
    assert scores["matching"] > scores["other"]
    assert scores["required"] == 1.0


def test_galet_assessor_uses_llm_api_without_temperature():
    class FakeGaletApi:
        def __init__(self):
            self.kwargs = None

        def create_response(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"output_text": '{"one": 0.9}'})()

    api = FakeGaletApi()
    result = GaletRelevanceAssessor(api, model="test-model").assess(
        "query", [_candidate("one", "text")]
    )
    assert result[0].relevance == 0.9
    assert api.kwargs["model"] == "test-model"
    assert "temperature" not in api.kwargs


def test_galet_assessor_rejects_invalid_output():
    class FakeGaletApi:
        def create_response(self, **kwargs):
            return type("Response", (), {"output_text": "not json"})()

    with pytest.raises(RelevanceAssessmentError):
        GaletRelevanceAssessor(FakeGaletApi(), model="test").assess(
            "query", [_candidate("one", "text")]
        )
