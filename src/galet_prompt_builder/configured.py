"""Own or borrow a configured memory runtime for prompt retrieval only."""
from dataclasses import replace
import logging

from galet_memory import build_memory, load_memory_config, ResourceLocation, DigestMatch
from .compiler import PromptCompiler
from .configuration import load_prompt_config, validate_policy
from .errors import PromptConfigurationError

logger = logging.getLogger(__name__)


class _ProfileSemanticMemory:
    def __init__(self, runtime):
        self.runtime = runtime

    def recall(self, request):
        settings = self.runtime.config
        profile = settings.embedding_profiles[settings.semantic.embedding_profile]
        return self.runtime.semantic.recall(replace(request, embedding_model=profile.model))


class _ConfiguredDigestMemory:
    """Read canonical digest locators; never publish digests or load snapshots."""
    def __init__(self, runtime):
        self.runtime = runtime

    def search_digests(self, *, account_name, query, count=3, session_id=None):
        runtime = self.runtime
        # Preserve a host's explicitly configured digest provider and filtering.
        if runtime.episodic.digest_search is not None:
            return runtime.episodic.search_digests(account_name=account_name, query=query,
                                                  count=count, session_id=session_id)
        if not count or not query.strip() or runtime.ingestion is None:
            return []
        if 'digests' not in runtime.ingestion.index.list_namespaces(account_name):
            return []
        settings = runtime.config
        profile = settings.embedding_profiles[settings.semantic.embedding_profile]
        vector = runtime.ingestion.embeddings.embed([query], model=profile.model)[0]
        matches = runtime.ingestion.index.query(account_name=account_name, namespaces=['digests'],
                                                 vector=vector, limit=count)
        results = []
        for match in matches:
            metadata = dict(match.record.metadata)
            resource = match.record.resource
            if settings.semantic.metadata_version == 3:
                if resource is None or resource.resource_type != 'digest':
                    continue
                sid, digest_id, sources = resource.session_id, resource.digest_id, resource.source_event_ids
                location = ResourceLocation(resource.external_root, resource.relative_path)
            else:
                sid = metadata.get('session_id') or match.record.source_id
                digest_id, sources = metadata.get('digest_id'), metadata.get('source_event_ids', ())
                location = None
            if session_id is not None and sid != session_id:
                continue
            if not runtime.episodic.is_digest_valid(account_name=account_name, session_id=sid,
                                                   digest_id=digest_id, source_event_ids=sources):
                continue
            try:
                path = (runtime.resolver.resolve(location) if location is not None else
                        runtime.resolver.authorize_path(metadata.get('path', '')))
                snippet = runtime.semantic.text_loader.load(path, max_chars=None)
            except (OSError, ValueError) as exc:
                logger.warning('prompt_digest_unavailable id=%s reason=%s', match.record.id, type(exc).__name__)
                continue
            if snippet.text.strip():
                results.append(DigestMatch(sid, snippet.text, float(match.score), snippet.truncated,
                    {**metadata, 'path': str(path), 'digest_id': digest_id, 'source_event_ids': list(sources)}))
        return results


class ConfiguredPromptBuilder:
    """Context-managed prompt compiler; session writes remain with the host."""
    def __init__(self, memory_runtime, policy, *, owns_memory=False, **compiler_options):
        self.memory = memory_runtime
        self.policy = validate_policy(policy)
        self.owns_memory = owns_memory
        self.closed = False
        self.compiler = PromptCompiler(
            semantic_memory=_ProfileSemanticMemory(memory_runtime) if memory_runtime.semantic else None,
            procedural_memory=memory_runtime.procedural,
            episodic_memory=memory_runtime.episodic,
            digest_memory=_ConfiguredDigestMemory(memory_runtime) if memory_runtime.episodic else None,
            **compiler_options,
        )

    def build(self, request, policy=None):
        if self.closed:
            raise RuntimeError('configured prompt builder is closed')
        # Empty namespaces opt into context selection, then all account namespaces.
        if request.include_semantic and self.memory.semantic and not request.semantic_namespaces:
            namespaces = []
            if self.memory.procedural and (request.context_name or request.skill_names):
                from galet_memory import ProceduralMemoryRequest
                context = self.memory.procedural.recall(ProceduralMemoryRequest(
                    account_name=request.account_name, context_name=request.context_name,
                    project_name=request.project_name, skill_names=request.skill_names,
                    create_if_missing=False))
                namespaces = list(context.search_namespaces)
            if not namespaces:
                namespaces = self.memory.semantic.list_namespaces(request.account_name)
            request = replace(request, semantic_namespaces=tuple(namespaces))
        return self.compiler.build(request, validate_policy(policy) if policy is not None else self.policy)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.owns_memory:
                self.memory.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('configured prompt builder is closed')
        return self

    def __exit__(self, *args):
        self.close()


def build_prompt_builder(config=None, *, path=None, base_dir=None, memory_runtime=None,
                         embedding_provider_factory=None, allow_create=True, **compiler_options):
    """Load prompt settings and construct or borrow one galet-memory runtime.

    Galet-memory owns store assembly and Galet owns credentials. Borrowed memory
    is never closed here. No module-global runtime, filesystem search or host
    application imports are used.
    """
    settings = load_prompt_config(config, path=path, base_dir=base_dir)
    if memory_runtime is not None:
        if settings.memory_config_path is not None or embedding_provider_factory is not None:
            raise PromptConfigurationError('borrowed memory cannot be combined with memory construction options')
        return ConfiguredPromptBuilder(memory_runtime, settings.policy, **compiler_options)
    memory_settings = load_memory_config(path=settings.memory_config_path)
    if not allow_create:
        memory_settings = replace(memory_settings, **{
            kind: replace(getattr(memory_settings, kind), database=replace(getattr(memory_settings, kind).database, create=False))
            for kind in ('semantic', 'episodic', 'working') if getattr(memory_settings, kind) is not None
        })
    runtime = build_memory(memory_settings, embedding_provider_factory=embedding_provider_factory)
    try:
        return ConfiguredPromptBuilder(runtime, settings.policy, owns_memory=True, **compiler_options)
    except BaseException:
        runtime.close()
        raise
