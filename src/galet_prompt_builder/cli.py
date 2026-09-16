from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--episodic-event-tokens", type=int, default=3000)
    parser.add_argument("--episodic-digest-tokens", type=int, default=1000)
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
        semantic_tokens=0,
        safety_margin_tokens=args.safety_margin_tokens,
    )


def _limits(args: argparse.Namespace) -> PromptLimits:
    return PromptLimits(
        maximum_total_tokens=max(args.total_tokens, 16000),
        maximum_procedural_tokens=1,
        maximum_episodic_event_tokens=max(
            args.episodic_event_tokens, 6000
        ),
        maximum_episodic_digest_tokens=max(
            args.episodic_digest_tokens, 2000
        ),
        maximum_semantic_tokens=1,
        maximum_events=20,
        maximum_digests=5,
        maximum_item_chars=12000,
    )


def _render_text(
    *,
    chat_name: str,
    session: EpisodicSession,
    database: Path,
    compiled,
) -> str:
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

    if not database.is_file():
        print(f"error: SQLite database not found: {database}", file=sys.stderr)
        return 2

    try:
        with SqliteEpisodicMemory(
            database,
            initialize_schema=False,
        ) as episodic_memory:
            session = _resolve_session(
                episodic_memory,
                account_name=args.account,
                friendly_name=args.chat_name,
            )
            compiled = PromptCompiler(
                episodic_memory=episodic_memory,
            ).compile(
                PromptRequest(
                    account_name=args.account,
                    conversation_id=session.session_id,
                    context_name=session.context_name or "",
                    current_input=args.request,
                    system_instructions=tuple(args.system),
                    include_procedural=False,
                    include_episodic=True,
                    include_semantic=False,
                    include_digests=True,
                ),
                _budgets(args),
                _limits(args),
            )
    except (OSError, RuntimeError, ValueError) as exc:
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
