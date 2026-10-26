from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from galet_memory import (
    Session,
    FileProceduralMemory,
    ProceduralLayout,
    SqliteEpisodicMemory,
)

from .compiler import PromptCompiler
from .contracts import PromptRequest
from .policy import PromptPolicy


DEFAULT_STORAGE_ROOT = Path("/home/junwin/lucy_storage")
DEFAULT_STORAGE_NAMESPACE = "data"
DEFAULT_ACCOUNT = "junwin"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a prompt from a request and an existing Lucy chat "
            "stored in SQLite."
        )
    )
    parser.add_argument("request", help="The current user request")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--chat-name", help="Exact friendly name of an existing session")
    selection.add_argument("--session-id", help="Existing session ID owned by the account")
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--config", type=Path, help="Prompt-builder configuration file")
    config.add_argument("--memory-config", type=Path, help="Shared galet-memory configuration file")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic recall")
    parser.add_argument("--no-digests", action="store_true", help="Disable cross-session digest search")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument(
        "--context-name",
        help="Project context to load; defaults to the chat's stored context name",
    )
    parser.add_argument(
        "--project-name", default="",
        help="Project scope when using the scoped procedural layout",
    )
    parser.add_argument(
        "--procedural-root", type=Path,
        help="Directory containing procedural contexts and skills (default: storage root/namespace)",
    )
    parser.add_argument(
        "--procedural-layout", choices=("lucy", "scoped"), default=None,
        help="Context and skill directory layout (default: lucy)",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help=f"Lucy storage root (default: {DEFAULT_STORAGE_ROOT})",
    )
    parser.add_argument(
        "--storage-namespace",
        default=None,
        help=f"Storage namespace (default: {DEFAULT_STORAGE_NAMESPACE})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Explicit chat2.sqlite path; overrides storage root and namespace",
    )
    parser.add_argument(
        "--namespaces",
        nargs="*",
        default=[],
        help=(
            "Semantic namespaces to search, for example: "
            "--namespaces vol_6 vol_7 documents (default: none)"
        ),
    )
    parser.add_argument(
        "--embedding-db",
        type=Path,
        help=(
            "Explicit embeddings-v2.sqlite path; overrides storage root "
            "and namespace"
        ),
    )
    parser.add_argument(
        "--credential-path",
        help="Directory containing Galet credential files",
    )
    parser.add_argument(
        "--sqlite-vec-extension",
        help="Explicit path to the sqlite-vec extension",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=[],
        help="System instruction; may be supplied more than once",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
    )
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--procedural-tokens", type=int, default=None)
    parser.add_argument("--episodic-event-tokens", type=int, default=None)
    parser.add_argument("--episodic-digest-tokens", type=int, default=None)
    parser.add_argument("--semantic-tokens", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-digests", type=int, default=None)
    parser.add_argument("--semantic-top-k", type=int, default=None)
    parser.add_argument(
        "--event-kinds",
        nargs="*",
        default=["user_message", "assistant_message", "session_digest"],
        help="Episodic event kinds to include",
    )
    parser.add_argument(
        "--include-structured-events",
        action="store_true",
        help="Include structured tool-call and operational event payloads",
    )
    parser.add_argument(
        "--semantic-score-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--digest-score-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--episodic-event-max-chars",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--episodic-digest-max-chars",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--semantic-item-max-chars",
        type=int,
        default=None,
    )
    parser.add_argument("--safety-margin-tokens", type=int, default=None)
    return parser


def _database_path(args: argparse.Namespace) -> Path:
    if args.db is not None:
        return args.db.expanduser()
    return (
        args.storage_root.expanduser()
        / args.storage_namespace
        / "chat2.sqlite"
    )


def _embedding_database_path(args: argparse.Namespace) -> Path:
    if args.embedding_db is not None:
        return args.embedding_db.expanduser()
    return (
        args.storage_root.expanduser()
        / args.storage_namespace
        / "embeddings-v2.sqlite"
    )


def _procedural_memory(args: argparse.Namespace) -> FileProceduralMemory:
    root = args.procedural_root or args.storage_root / args.storage_namespace
    layout = ProceduralLayout.lucy() if args.procedural_layout == "lucy" else ProceduralLayout()
    return FileProceduralMemory(root.expanduser(), layout)


def _semantic_memory(
    args: argparse.Namespace,
    database: Path,
    resources: ExitStack,
) -> tuple[Any, Any]:
    from galet.embedding_router import EmbeddingRouter
    from galet.mistral_embedding import MistralEmbeddingApi
    from galet.openai_embedding import OpenAIEmbeddingApi
    from galet.settings import Settings
    from galet_memory import EmbeddingDigestRecall, VectorSemanticMemory
    from galet_memory.galet_adapter import GaletEmbeddingProvider
    from galet_memory.ports import FileTextLoader, SqliteVecEmbeddingIndex

    settings = Settings(credential_path=args.credential_path)
    embeddings = GaletEmbeddingProvider(
        EmbeddingRouter(
            openai_api=OpenAIEmbeddingApi(settings=settings),
            mistral_api=MistralEmbeddingApi(settings=settings),
        )
    )
    index = resources.enter_context(
        SqliteVecEmbeddingIndex(
            database,
            sqlite_vec_extension_path=args.sqlite_vec_extension,
            initialize_schema=False,
        )
    )
    loader = FileTextLoader()
    semantic = VectorSemanticMemory(
        embeddings=embeddings,
        index=index,
        text_loader=loader,
    )
    return semantic, EmbeddingDigestRecall(
        embeddings=embeddings, index=index, text_loader=loader,
    )


def _resolve_session(
    memory: SqliteEpisodicMemory,
    *,
    account_name: str,
    friendly_name: str,
) -> Session:
    expected = friendly_name.strip().casefold()
    matches = []
    cursor = None
    while True:
        page = memory.list_sessions(account_name=account_name, count=100, cursor=cursor)
        matches.extend(session for session in page.items
                       if (session.friendly_name or "").strip().casefold() == expected)
        cursor = page.next_cursor
        if cursor is None:
            break
    if not matches:
        raise ValueError(
            f"no chat named {friendly_name!r} for account {account_name!r}"
        )
    if len(matches) > 1:
        session_ids = ", ".join(session.session_id for session in matches)
        raise ValueError(
            f"chat name {friendly_name!r} is ambiguous; "
            f"matching session IDs: {session_ids}"
        )
    return matches[0]


def _render_text(
    *,
    chat_name: str,
    session: Session,
    database: Path,
    compiled,
) -> str:
    candidate_rows = []
    for item in compiled.metrics.candidates:
        status = "selected" if item.selected else item.drop_reason
        tokens = f"{item.final_tokens}/{item.original_tokens}"
        candidate_rows.append(
            f"{item.source:<18} {item.relevance:>6.3f} "
            f"{status:<27} {tokens:>11}  {item.candidate_id}"
        )
    candidates = (
        "source              score status                      tokens  id\n"
        + "\n".join(candidate_rows)
        if candidate_rows
        else "No memory candidates were retrieved."
    )
    metrics = json.dumps(
        asdict(compiled.metrics),
        indent=2,
        ensure_ascii=False,
    )
    return (
        f"Chat: {chat_name}\n"
        f"Session: {session.session_id}\n"
        f"Database: {database}\n\n"
        "--- Prompt ---\n"
        f"{compiled.text}\n\n"
        "--- Candidate decisions ---\n"
        f"{candidates}\n\n"
        "--- Metrics ---\n"
        f"{metrics}"
    )


def _render_json(
    *,
    chat_name: str,
    session: Session,
    database: Path,
    compiled,
) -> str:
    return json.dumps(
        {
            "chat_name": chat_name,
            "session_id": session.session_id,
            "context_name": session.context_name or "",
            "database": str(database),
            "messages": compiled.provider_messages,
            "metrics": asdict(compiled.metrics),
        },
        indent=2,
        ensure_ascii=False,
    )


_POLICY_ARGS = {
    'total_tokens': 'total_tokens', 'safety_margin_tokens': 'safety_margin_tokens',
    'procedural_tokens': 'procedural_tokens', 'episodic_event_tokens': 'episodic_event_tokens',
    'episodic_digest_tokens': 'episodic_digest_tokens', 'semantic_tokens': 'semantic_tokens',
    'max_events': 'maximum_events', 'max_digests': 'maximum_digests',
    'semantic_top_k': 'maximum_semantic_documents',
    'semantic_score_threshold': 'semantic_score_threshold', 'digest_score_threshold': 'digest_score_threshold',
    'episodic_event_max_chars': 'episodic_event_max_chars',
    'episodic_digest_max_chars': 'episodic_digest_max_chars', 'semantic_item_max_chars': 'semantic_item_max_chars',
}


def _select_session(args, memory):
    if args.session_id:
        session = memory.get_session(account_name=args.account, session_id=args.session_id)
        if session is None:
            raise ValueError('session not found for account')
        return session
    return _resolve_session(memory, account_name=args.account, friendly_name=args.chat_name)


def _configured_main(args):
    from .configured import build_prompt_builder
    from .errors import PromptConfigurationError

    conflicts = ('db', 'embedding_db', 'storage_root', 'storage_namespace', 'procedural_root',
                 'procedural_layout', 'sqlite_vec_extension', 'credential_path')
    if any(getattr(args, key) is not None for key in conflicts):
        raise PromptConfigurationError('shared configuration cannot be combined with legacy storage/provider flags')
    direct = ({'config_version': 1, 'memory_config_path': str(args.memory_config.expanduser().resolve())}
              if args.memory_config is not None else None)
    with build_prompt_builder(direct, path=args.config, allow_create=False) as builder:
        runtime = builder.memory
        if runtime.episodic is None:
            raise PromptConfigurationError('the comparison CLI requires configured episodic memory')
        session = _select_session(args, runtime.episodic)
        context = args.context_name if args.context_name is not None else session.context_name or ''
        if args.context_name and (runtime.procedural is None or
                runtime.procedural.repository.resolve(args.account, context, args.project_name) is None):
            raise PromptConfigurationError('requested procedural context was not found')
        policy = replace(builder.policy, **{field: getattr(args, key) for key, field in _POLICY_ARGS.items()
                                            if getattr(args, key) is not None})
        compiled = builder.build(PromptRequest(
            account_name=args.account, conversation_id=session.session_id,
            context_name=context, project_name=args.project_name, current_input=args.request,
            system_instructions=tuple(args.system), semantic_namespaces=tuple(args.namespaces or ()),
            episodic_event_kinds=tuple(args.event_kinds),
            include_structured_episodic_events=args.include_structured_events,
            include_procedural=bool(context), include_episodic=True,
            include_semantic=not args.no_semantic, include_digests=not args.no_digests,
        ), policy)
        database = Path(runtime.episodic.db_path)
    render = _render_json if args.output_format == 'json' else _render_text
    print(render(chat_name=args.chat_name or session.friendly_name or session.session_id,
                 session=session, database=database, compiled=compiled))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = args.config is not None or args.memory_config is not None or any(
        name in os.environ for name in ('GALET_PROMPT_BUILDER_CONFIG_PATH', 'GALET_MEMORY_CONFIG_PATH'))
    if selected:
        try:
            return _configured_main(args)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    args.storage_root = args.storage_root or DEFAULT_STORAGE_ROOT
    args.storage_namespace = args.storage_namespace or DEFAULT_STORAGE_NAMESPACE
    args.procedural_layout = args.procedural_layout or 'lucy'
    defaults = PromptPolicy()
    for key, field in _POLICY_ARGS.items():
        if getattr(args, key) is None:
            setattr(args, key, getattr(defaults, field))
    database = _database_path(args)
    embedding_database = _embedding_database_path(args)

    if not database.is_file():
        print(f"error: SQLite database not found: {database}", file=sys.stderr)
        return 2
    if args.namespaces and not embedding_database.is_file():
        print(
            f"error: embedding SQLite database not found: {embedding_database}",
            file=sys.stderr,
        )
        return 2

    try:
        with ExitStack() as resources:
            semantic_memory, digest_recall = (
                _semantic_memory(args, embedding_database, resources)
                if embedding_database.is_file() else (None, None)
            )
            episodic_memory = resources.enter_context(
                SqliteEpisodicMemory(
                    database,
                    initialize_schema=False,
                    digest_search=digest_recall,
                )
            )
            session = _select_session(args, episodic_memory)
            context_name = args.context_name if args.context_name is not None else (session.context_name or "")
            procedural_memory = _procedural_memory(args)
            if args.context_name and procedural_memory.repository.resolve(args.account, context_name, args.project_name) is None:
                raise ValueError(f"project context {context_name!r} not found under {procedural_memory.repository.root}")
            compiled = PromptCompiler(
                episodic_memory=episodic_memory,
                digest_memory=episodic_memory if digest_recall is not None else None,
                semantic_memory=semantic_memory if args.namespaces else None,
                procedural_memory=procedural_memory,
            ).build(
                PromptRequest(
                    account_name=args.account,
                    conversation_id=session.session_id,
                    context_name=context_name,
                    project_name=args.project_name,
                    current_input=args.request,
                    system_instructions=tuple(args.system),
                    semantic_namespaces=tuple(args.namespaces),
                    episodic_event_kinds=tuple(args.event_kinds),
                    include_structured_episodic_events=(
                        args.include_structured_events
                    ),
                    include_procedural=bool(context_name and args.procedural_tokens),
                    include_episodic=True,
                    include_semantic=bool(args.namespaces) and not args.no_semantic,
                    include_digests=not args.no_digests,
                ),
                PromptPolicy(
                    total_tokens=args.total_tokens,
                    safety_margin_tokens=args.safety_margin_tokens,
                    procedural_tokens=args.procedural_tokens,
                    episodic_event_tokens=args.episodic_event_tokens,
                    episodic_digest_tokens=args.episodic_digest_tokens,
                    semantic_tokens=args.semantic_tokens,
                    maximum_events=args.max_events,
                    maximum_digests=args.max_digests,
                    maximum_semantic_documents=max(args.semantic_top_k, 1),
                    semantic_score_threshold=args.semantic_score_threshold,
                    digest_score_threshold=args.digest_score_threshold,
                    episodic_event_max_chars=args.episodic_event_max_chars,
                    episodic_digest_max_chars=args.episodic_digest_max_chars,
                    semantic_item_max_chars=args.semantic_item_max_chars,
                ),
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        output = _render_json(
            chat_name=args.chat_name or session.friendly_name or session.session_id,
            session=session,
            database=database,
            compiled=compiled,
        )
    else:
        output = _render_text(
            chat_name=args.chat_name or session.friendly_name or session.session_id,
            session=session,
            database=database,
            compiled=compiled,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
