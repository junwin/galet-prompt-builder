from galet_memory import DigestMatch

from galet_prompt_builder import PromptCompiler, PromptPolicy, PromptRequest


class Episodic:
    def search_digests(self, **kwargs):
        return [DigestMatch("old", "Relevant digest", 0.5)]


def test_policy_build_owns_budgets_limits_and_thresholds():
    policy = PromptPolicy(
        total_tokens=300, safety_margin_tokens=20,
        procedural_tokens=0, episodic_event_tokens=0,
        episodic_digest_tokens=80, semantic_tokens=0,
        digest_score_threshold=0.6,
    )
    compiled = PromptCompiler(digest_memory=Episodic()).build(
        PromptRequest(account_name="alice", current_input="Find an earlier digest",
                      include_semantic=False), policy,
    )
    assert compiled.metrics.episodic_digests.budget_tokens == 80
    assert compiled.metrics.semantic.budget_tokens == 0
    assert compiled.metrics.episodic_digests.selected_items == 0
    assert compiled.metrics.candidates[0].drop_reason == "below_relevance_threshold"
