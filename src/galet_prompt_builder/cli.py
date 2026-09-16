from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from galet_memory import (
    EpisodicSession,
    EpisodicSessionQuery,
    SqliteEpisodicMemory,
)

from .compiler import PromptCompiler
from .contracts import PromptBudgets, PromptLimits, PromptRequest


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
    parser.add_argument(
        "--chat-name",
        required=True,
        help="Exact friendly name of the existing chat session",
    )
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help=f"Lucy storage root (default: {DEFAULT_STORAGE_ROOT})",
    )
    parser.add_argument(
        "--storage-namespace",
        default=DEFAULT_STORAGE_NAMESPACE,
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
    parser.add_argument("--total-tokens", type=int, default=8000)
    parser.add_argument("--episodic-event-tokens", type=int, default=1000)
    parser.add_argument("--episodic-digest-tokens", type=int, default=500)
    parser.add_argument("--semantic-tokens", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=6)
    parser.add_argument("--max-digests", type=int, default=2)
    parser.add_argument("--semantic-top-k", type=int, default=3)
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
        default=0.30,
    )
    parser.add_argument(
        "--digest-score-threshold",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--episodic-event-max-chars",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--episodic-digest-max-chars",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--semantic-item-max-chars",
        type=int,
        default=1800,
    )
    parser.add_argument("--safety-margin-tokens", type=int, default=500)
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


def _semantic_memory(
    args: argparse.Namespace,
    database: Path,
    resources: ExitStack,
) -> Any:
    from galet.embedding_router import EmbeddingRouter
    from galet.mistral_embedding import MistralEmbeddingApi
    from galet.openai_embedding import OpenAIEmbeddingApi
    from galet.settings import Settings
    from galet_memory import VectorSemanticMemory
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
    return VectorSemanticMemory(
        embeddings=embeddings,
        index=index,
        text_loader=FileTextLoader(),
    )


def _resolve_session(
    memory: SqliteEpisodicMemory,
    *,
    account_name: str,
    friendly_name: str,
) -> EpisodicSession:
    expected = friendly_name.strip().casefold()
    sessions = memory.list_sessions(
        EpisodicSessionQuery(account_name=account_name, limit=10000)
    )
    matches = [
        session
        for session in sessions
        if (session.friendly_name or "").strip().casefold() == expected
    ]
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


def _budgets(args: argparse.Namespace) -> PromptBudgets:
    return PromptBudgets(
        total_tokens=args.total_tokens,
        procedural_tokens=0,
        episodic_event_tokens=args.episodic_event_tokens,
        episodic_digest_tokens=args.episodic_digest_tokens,
        semantic_tokens=args.semantic_tokens if args.namespaces else 0,
        safety_margin_tokens=args.safety_margin_tokens,
    )


def _limits(args: argparse.Namespace) -> PromptLimits:
    return PromptLimits(
        maximum_total_tokens=args.total_tokens,
        maximum_procedural_tokens=1,
        maximum_episodic_event_tokens=max(args.episodic_event_tokens, 1),
        maximum_episodic_digest_tokens=max(
            args.episodic_digest_tokens, 1
        ),
        maximum_semantic_tokens=max(args.semantic_tokens, 1),
        maximum_events=args.max_events,
        maximum_digests=args.max_digests,
        maximum_semantic_documents=max(args.semantic_top_k, 1),
        maximum_item_chars=12000,
        maximum_episodic_event_chars=args.episodic_event_max_chars,
        maximum_episodic_digest_chars=args.episodic_digest_max_chars,
        maximum_semantic_item_chars=args.semantic_item_max_chars,
    )


def _render_text(
    *,
    chat_name: str,
    session: EpisodicSession,
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
    session: EpisodicSession,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
            episodic_memory = resources.enter_context(
                SqliteEpisodicMemory(
                    database,
                    initialize_schema=False,
                )
            )
            session = _resolve_session(
                episodic_memory,
                account_name=args.account,
                friendly_name=args.chat_name,
            )
            semantic_memory = (
                _semantic_memory(args, embedding_database, resources)
                if args.namespaces
                else None
            )
            compiled = PromptCompiler(
                episodic_memory=episodic_memory,
                semantic_memory=semantic_memory,
            ).compile(
                PromptRequest(
                    account_name=args.account,
                    conversation_id=session.session_id,
                    context_name=session.context_name or "",
                    current_input=args.request,
                    system_instructions=tuple(args.system),
                    semantic_namespaces=tuple(args.namespaces),
                    semantic_score_threshold=(
                        args.semantic_score_threshold
                    ),
                    episodic_event_kinds=tuple(args.event_kinds),
                    include_structured_episodic_events=(
                        args.include_structured_events
                    ),
                    episodic_digest_score_threshold=(
                        args.digest_score_threshold
                    ),
                    include_procedural=False,
                    include_episodic=True,
                    include_semantic=bool(args.namespaces),
                    include_digests=True,
                ),
                _budgets(args),
                _limits(args),
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        output = _render_json(
            chat_name=args.chat_name,
            session=session,
            database=database,
            compiled=compiled,
        )
    else:
        output = _render_text(
            chat_name=args.chat_name,
            session=session,
            database=database,
            compiled=compiled,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
