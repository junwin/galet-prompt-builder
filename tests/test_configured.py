import json
from dataclasses import replace
from pathlib import Path

import pytest
from galet_memory import build_memory, NewEvent, CurationService
from galet_memory.ports import StoredEmbedding, EmbeddingResource
from galet_prompt_builder import (build_prompt_builder, load_prompt_config, PromptConfigurationError,
                                  PromptRequest, PromptPolicy, MemoryRetrievalError)


class Embeddings:
    calls = []
    def __init__(self, profile):
        self.profile = profile

    def embed(self, texts, *, model):
        assert model == self.profile.model
        self.calls.append(model)
        return [[1.0] + [0.0] * 1535 for _ in texts]


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.delenv('GALET_PROMPT_BUILDER_CONFIG_PATH', raising=False)
    monkeypatch.delenv('GALET_MEMORY_CONFIG_PATH', raising=False)
    import galet_memory.assembly as assembly
    monkeypatch.setattr(assembly, 'galet_embedding_factory', lambda: Embeddings)
    Embeddings.calls = []
    data = tmp_path / 'data'
    data.mkdir()
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'orchard.md').write_text('Orchard apples grow by a quiet garden wall.')
    memory = {'config_version': 1, 'external_roots': {'data': str(data), 'notes': str(notes)},
        'embedding_profiles': {'chosen': {'provider': 'test', 'model': 'configured-model', 'dimensions': 1536}},
        'memory': {
            'semantic': {'metadata_version': 3, 'embedding_profile': 'chosen',
                'database': {'external_root': 'data', 'relative_path': 'semantic.sqlite', 'create': True}},
            'episodic': {'database': {'external_root': 'data', 'relative_path': 'episodic.sqlite', 'create': True}},
            'procedural': {'root': {'external_root': 'data', 'relative_path': '.'}},
            'working': {'database': {'external_root': 'data', 'relative_path': 'working.sqlite', 'create': True}},
        }}
    memory_path = tmp_path / 'memory.json'
    memory_path.write_text(json.dumps(memory))
    prompt_path = tmp_path / 'prompt.json'
    prompt_path.write_text(json.dumps({'config_version': 1, 'memory_config_path': 'memory.json',
        'policy': {'total_tokens': 6000, 'semantic_score_threshold': 0.25}}))
    with build_memory(memory) as runtime:
        runtime.ingest_file(account_name='john', external_root='notes', relative_path='orchard.md', namespace='books')
        runtime.episodic.create_session(account_name='john', session_id='chat', friendly_name='Reading', context_name='reading')
        runtime.episodic.append_event(account_name='john', session_id='chat',
            event=NewEvent('user', 'Earlier orchard question', 'john', correlation_ids=('original',)))
        runtime.episodic.create_session(account_name='other', session_id='foreign')
        runtime.procedural.repository.save_skill(account_name='john', skill_name='reading', text='Be precise about sources.')
        runtime.procedural.repository.save_context(account_name='john', context_name='reading',
            text='Remember the garden.', frontmatter={'imports': ['reading'], 'search_namespaces': ['books']})
    return prompt_path, memory_path, notes


def request(**overrides):
    values = dict(account_name='john', current_input='What grows in the orchard?', conversation_id='chat',
                  context_name='reading', semantic_namespaces=())
    values.update(overrides)
    return PromptRequest(**values)


def test_configured_builder_reads_all_prompt_sources_and_selected_model(files):
    prompt, _, _ = files
    with build_prompt_builder(path=prompt, allow_create=False) as builder:
        assert builder.policy.total_tokens == 6000
        result = builder.build(request())
        assert 'Orchard apples' in result.text
        assert 'Earlier orchard question' in result.text
        assert 'Remember the garden' in result.text
        assert 'Be precise about sources' in result.text
        assert result.metrics.semantic.selected_items == 1
        assert result.metrics.total_used_tokens <= 5500
        assert Embeddings.calls and all(model == 'configured-model' for model in Embeddings.calls)
        runtime = builder.memory
    assert builder.closed
    with pytest.raises(RuntimeError, match='closed'):
        runtime.recall(account_name='john', query='orchard')
    with pytest.raises(RuntimeError, match='closed'):
        builder.build(request())


def test_provider_failure_cannot_masquerade_as_empty_semantic_memory(files, monkeypatch, caplog):
    prompt, _, _ = files
    with build_prompt_builder(path=prompt, allow_create=False) as builder:
        def unavailable(*args, **kwargs):
            raise RuntimeError('Missing credentials')
        monkeypatch.setattr(builder.memory.ingestion.embeddings, 'embed', unavailable)
        # galet-memory keeps its keyword fallback, with an explicit failure reason.
        fallback = builder.memory.recall(account_name='john', query='orchard', namespaces=['books'])
        assert fallback.metadata['reason'] == 'embedding_provider_unavailable'
        with pytest.raises(MemoryRetrievalError, match='GALET_CREDENTIAL_PATH'):
            builder.build(request(include_digests=False))
        assert 'prompt_semantic_recall_failed' in caplog.text


def test_successful_empty_semantic_search_is_not_a_provider_failure(files, caplog):
    prompt, _, _ = files
    import logging
    with caplog.at_level(logging.INFO), build_prompt_builder(path=prompt, allow_create=False) as builder:
        result = builder.build(request(semantic_namespaces=('empty',), include_digests=False))
    assert result.metrics.semantic.retrieved_items == 0
    assert 'prompt_semantic_recall_done' in caplog.text
    assert 'documents=0' in caplog.text
    assert 'prompt_compile_done' in caplog.text


def test_borrowed_runtime_remains_usable_and_respects_explicit_namespaces(files):
    _, memory, _ = files
    import galet_memory
    with galet_memory.build_memory(path=memory) as runtime:
        with build_prompt_builder({'config_version': 1}, memory_runtime=runtime) as builder:
            assert not builder.owns_memory
            compiled = builder.build(request(semantic_namespaces=('missing',)))
            assert compiled.metrics.semantic.selected_items == 0
        assert runtime.recall(account_name='john', query='orchard', namespaces=['books']).documents
    with pytest.raises(PromptConfigurationError, match='borrowed memory'):
        build_prompt_builder({'config_version': 1, 'memory_config_path': str(memory)}, memory_runtime=runtime)


def test_config_precedence_relative_paths_and_bad_sources(files, monkeypatch):
    prompt, memory, _ = files
    monkeypatch.setenv('GALET_PROMPT_BUILDER_CONFIG_PATH', str(prompt))
    assert load_prompt_config().memory_config_path == memory
    assert load_prompt_config({'config_version': 1}).memory_config_path is None
    with pytest.raises(PromptConfigurationError, match='requires base_dir'):
        load_prompt_config({'config_version': 1, 'memory_config_path': 'memory.json'})
    assert load_prompt_config({'config_version': 1, 'memory_config_path': 'memory.json'}, base_dir=prompt.parent).memory_config_path == memory
    with pytest.raises(PromptConfigurationError, match='cannot load'):
        load_prompt_config(path=prompt.parent / 'missing.json')
    monkeypatch.delenv('GALET_PROMPT_BUILDER_CONFIG_PATH')
    monkeypatch.setenv('GALET_MEMORY_CONFIG_PATH', str(memory))
    with build_prompt_builder(allow_create=False) as builder:
        assert builder.build(request()).metrics.semantic.selected_items == 1


@pytest.mark.parametrize('policy', [
    {'unknown': 1}, {'total_tokens': True}, {'maximum_events': -1},
    {'semantic_score_threshold': 1.5}, {'total_tokens': '6000'},
])
def test_invalid_policy_fails_before_store_construction(policy, monkeypatch):
    import galet_prompt_builder.configured as module
    monkeypatch.setattr(module, 'build_memory', lambda *a, **kw: pytest.fail('must validate policy first'))
    with pytest.raises(PromptConfigurationError):
        build_prompt_builder({'config_version': 1, 'policy': policy})


def test_source_deletion_and_account_isolation(files):
    prompt, _, notes = files
    with build_prompt_builder(path=prompt) as builder:
        unrelated = builder.build(request(account_name='other', conversation_id='foreign', context_name=''))
        assert 'Orchard apples' not in unrelated.text
        assert 'Earlier orchard question' not in unrelated.text
        (notes / 'orchard.md').unlink()
        missing = builder.build(request())
        assert missing.metrics.semantic.selected_items == 0
        with pytest.raises(MemoryRetrievalError):
            builder.build(request(conversation_id='foreign'))


def test_read_existing_mode_cannot_create_missing_stores(files):
    prompt, _, _ = files
    (prompt.parent / 'data/working.sqlite').unlink()
    with pytest.raises(FileNotFoundError):
        build_prompt_builder(path=prompt, allow_create=False)
    assert not (prompt.parent / 'data/working.sqlite').exists()


def test_builder_failure_closes_owned_runtime(files, monkeypatch):
    import galet_prompt_builder.configured as module
    prompt, _, _ = files
    original = module.build_memory
    opened = []
    def build(*args, **kwargs):
        runtime = original(*args, **kwargs)
        opened.append(runtime)
        return runtime
    monkeypatch.setattr(module, 'build_memory', build)
    with pytest.raises(TypeError):
        module.build_prompt_builder(path=prompt, unknown_compiler_option=True)
    with pytest.raises(RuntimeError, match='closed'):
        opened[0].recall(account_name='john', query='orchard')


def test_digest_locators_are_checked_and_invalidated_digest_is_not_reintroduced(files):
    prompt, memory, _ = files
    with build_memory(path=memory) as runtime:
        class Generator:
            def generate(self, request):
                return 'Archive of the orchard conversation.'
        archived = CurationService(runtime.episodic, Generator()).archive(account_name='john', session_id='chat')
        text = archived.digest
        digest_path = prompt.parent / 'data/digest.md'
        digest_path.write_text(text)
        runtime.ingestion.index.upsert(StoredEmbedding(
            id='digest', account_name='john', namespace='digests', vector=[1.0]+[0.0]*1535,
            source_type='digest', source_id='chat', model='configured-model', provider='test',
            metadata={'path': '/wrong/legacy/path.md'}, keywords=['orchard'],
            resource=EmbeddingResource('digest', 'data', 'digest.md', session_id='chat',
                digest_id=archived.boundary_event.event_id, source_event_ids=archived.source_event_ids,
                representation='digest')))
        runtime.episodic.create_session(account_name='john', session_id='target')
    with build_prompt_builder(path=prompt) as builder:
        compiled = builder.build(request(conversation_id='target'))
        assert compiled.metrics.episodic_digests.selected_items == 1
        assert text in compiled.text
        builder.memory.episodic.invalidate_exchange(account_name='john', session_id='chat', correlation_id='original')
        assert text not in builder.build(request(conversation_id='target')).text
    with build_prompt_builder(path=prompt) as reopened:
        assert text not in reopened.build(request(conversation_id='target')).text


def test_configured_cli_uses_file_policy_and_session_id(files, capsys):
    from galet_prompt_builder.cli import main
    prompt, _, _ = files
    assert main(['orchard', '--config', str(prompt), '--account', 'john', '--session-id', 'chat', '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['database'] == str(prompt.parent / 'data/episodic.sqlite')
    assert payload['metrics']['total_limit_tokens'] == 6000
    assert 'Orchard apples' in json.dumps(payload['messages'])
    assert main(['orchard', '--memory-config', str(files[1]), '--account', 'john', '--chat-name', 'Reading',
                 '--total-tokens', '7000', '--format', 'json']) == 0
    assert json.loads(capsys.readouterr().out)['metrics']['total_limit_tokens'] == 7000


def test_configured_cli_rejects_legacy_overrides_and_foreign_session(files, capsys):
    from galet_prompt_builder.cli import main
    prompt, _, _ = files
    assert main(['q', '--config', str(prompt), '--account', 'john', '--session-id', 'chat', '--db', '/legacy.sqlite']) == 2
    assert 'legacy' in capsys.readouterr().err
    assert main(['q', '--config', str(prompt), '--account', 'john', '--session-id', 'foreign']) == 2
    assert 'session not found for account' in capsys.readouterr().err


def test_borrowed_runtime_preserves_host_digest_search(files):
    from galet_memory import DigestMatch
    _, memory, _ = files
    with build_memory(path=memory) as runtime:
        class Generator:
            def generate(self, request):
                return 'The host owns this archived summary.'
        archive = CurationService(runtime.episodic, Generator()).archive(account_name='john', session_id='chat')
        runtime.episodic.create_session(account_name='john', session_id='target')
        calls = []
        def host_search(search):
            calls.append(search)
            return [DigestMatch('chat', archive.digest, 0.9, metadata={'digest_id': archive.boundary_event.event_id})]
        runtime.episodic.digest_search = host_search
        with build_prompt_builder({'config_version': 1}, memory_runtime=runtime) as builder:
            compiled = builder.build(request(conversation_id='target', include_semantic=False))
            assert archive.digest in compiled.text
        assert runtime.episodic.digest_search is host_search
        assert calls[0].account_name == 'john'


def test_sample_config_loads_and_duplicate_keys_are_rejected(files):
    sample = Path(__file__).parents[1] / 'galet_prompt_builder_sample.json'
    config = load_prompt_config(path=sample)
    assert config.policy.total_tokens == 6000
    assert config.memory_config_path.name == 'galet-memory-mcp.json'
    prompt, _, _ = files
    prompt.write_text('{"config_version":1,"config_version":1}')
    with pytest.raises(PromptConfigurationError):
        load_prompt_config(path=prompt)


def test_cli_schema_mismatch_does_not_fall_back_to_legacy(files, capsys):
    from galet_prompt_builder.cli import main
    prompt, memory, _ = files
    raw = json.loads(memory.read_text())
    raw['memory']['semantic']['metadata_version'] = 2
    memory.write_text(json.dumps(raw))
    assert main(['q', '--config', str(prompt), '--account', 'john', '--session-id', 'chat']) == 2
    assert 'metadata version mismatch' in capsys.readouterr().err
