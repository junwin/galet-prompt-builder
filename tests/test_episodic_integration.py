"""Prompt selection against the merged explicit SQLite and JSONL contracts."""
import pytest
from galet_memory import SqliteEpisodicMemory, JsonlEpisodicMemory, NewEvent, CurationService, DigestMatch
from galet_prompt_builder import PromptCompiler, PromptRequest, PromptPolicy, MemoryRetrievalError


@pytest.fixture(params=[SqliteEpisodicMemory, JsonlEpisodicMemory])
def memory(request, tmp_path):
    with request.param(tmp_path / 'store') as store:
        store.create_session(account_name='acct', session_id='s')
        yield store


def build(memory, *, digest_memory=None, kinds=None, max_events=10, **kwargs):
    return PromptCompiler(episodic_memory=memory, digest_memory=digest_memory).build(
        PromptRequest(account_name='acct', conversation_id='s', current_input='Continue',
                      episodic_event_kinds=kinds, include_procedural=False, include_semantic=False),
        PromptPolicy(total_tokens=5000, safety_margin_tokens=100, procedural_tokens=0,
                     episodic_event_tokens=2500, episodic_digest_tokens=1000,
                     semantic_tokens=0, maximum_events=max_events, **kwargs),
    )


def append(memory, content, role='user', actor='acct', correlation_ids=(), kind=''):
    return memory.append_event(account_name='acct', session_id='s',
                               event=NewEvent(role, content, actor, kind=kind, correlation_ids=correlation_ids))


def event_messages(compiled):
    return [message for message in compiled.messages if message.source == 'episodic_event']


def test_filters_apply_before_count_and_multiple_actors_keep_roles(memory):
    append(memory, 'Older')
    append(memory, {'text': 'Full payload', 'attachments': [{'id': 'photo-1'}]}, correlation_ids=('c',))
    for index in range(5):
        append(memory, f'Report {index}', role='system', actor='processor', kind='prompt_report')
    append(memory, 'Newer')
    messages = event_messages(build(memory, kinds=('user_message',), max_events=2))
    assert len(messages) == 2 and 'Full payload' in messages[0].content and 'photo-1' in messages[0].content
    assert messages[1].content == 'Newer'
    append(memory, 'Peace answer', role='assistant', actor='peace')
    append(memory, 'Belle answer', role='assistant', actor='belle')
    messages = event_messages(build(memory, kinds=('assistant_message',), max_events=2))
    assert [(m.role, m.content) for m in messages] == [('assistant', 'Peace answer'), ('assistant', 'Belle answer')]


def test_archive_boundary_reset_invalidation_and_digest_search(memory):
    target = append(memory, 'Private question', correlation_ids=('c',))
    append(memory, 'Private answer', role='assistant', actor='peace', correlation_ids=('c',))
    class Generator:
        def generate(self, request):
            return 'Private archive summary'
    curation = CurationService(memory, Generator())
    archive = curation.archive(account_name='acct', session_id='s')
    append(memory, 'Fresh question')
    requests = []
    def search(request):
        requests.append(request)
        return [DigestMatch('s', archive.digest, score=0.9,
                            metadata={'digest_id': archive.boundary_event.event_id})]
    memory.digest_search = search
    compiled = build(memory, digest_memory=memory)
    assert [m.content for m in event_messages(compiled)] == ['Private archive summary', 'Fresh question']
    assert compiled.metrics.episodic_digests.retrieved_items == 0  # deduplicated active boundary
    assert requests[-1].account_name == 'acct'
    curation.reset_context(account_name='acct', session_id='s')
    assert event_messages(build(memory)) == []
    memory.invalidate_exchange(account_name='acct', session_id='s', correlation_id='c')
    compiled = build(memory, digest_memory=memory)
    assert not any('Private' in m.content for m in compiled.messages)
    assert target.event_id in [e.event_id for e in memory.get_audit_snapshot(account_name='acct', session_id='s').events]


def test_explicit_digest_only_mode_zero_limits_and_account_ownership(memory):
    append(memory, 'Hidden by zero event limit')
    assert event_messages(build(memory, max_events=0, maximum_digests=0)) == []
    memory.create_session(account_name='other', session_id='foreign')
    with pytest.raises(MemoryRetrievalError):
        PromptCompiler(episodic_memory=memory).build(
            PromptRequest(account_name='acct', conversation_id='foreign', current_input='Read it'),
            PromptPolicy(),
        )
    class Search:
        def search_digests(self, **kwargs):
            assert kwargs['account_name'] == 'acct'
            return [DigestMatch('old', 'Earlier decision', score=1)]
    compiled = PromptCompiler(digest_memory=Search()).build(
        PromptRequest(account_name='acct', current_input='Find earlier decisions'), PromptPolicy()
    )
    assert any('Earlier decision' in m.content for m in compiled.messages)
