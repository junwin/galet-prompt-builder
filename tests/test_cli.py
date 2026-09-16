import json

from galet_memory import (
    EpisodicEvent,
    SemanticDocument,
    SemanticMemoryResult,
    SqliteEpisodicMemory,
)

import galet_prompt_builder.cli as cli_module
from galet_prompt_builder.cli import main


def _database(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    database = data / "chat2.sqlite"
    with SqliteEpisodicMemory(database) as memory:
        session = memory.create_session(
            account_name="junwin",
            agent_name="nelly",
            session_id="session-123",
            friendly_name="Prompt Comparison",
            context_name="lucyproject",
        )
        memory.add_events(
            session.session_id,
            [
                EpisodicEvent("user", "Earlier question"),
                EpisodicEvent("assistant", "Earlier answer"),
            ],
        )
    return database


def test_cli_resolves_friendly_name_and_compiles_prompt(tmp_path, capsys):
    _database(tmp_path)

    result = main(
        [
            "What should happen next?",
            "--chat-name",
            "Prompt Comparison",
            "--storage-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Session: session-123" in captured.out
    assert "Earlier question" in captured.out
    assert "Earlier answer" in captured.out
    assert "What should happen next?" in captured.out


def test_cli_json_output_contains_provider_messages(tmp_path, capsys):
    database = _database(tmp_path)

    result = main(
        [
            "Compare this prompt",
            "--chat-name",
            "prompt comparison",
            "--db",
            str(database),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["chat_name"] == "prompt comparison"
    assert payload["session_id"] == "session-123"
    assert payload["context_name"] == "lucyproject"
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "Compare this prompt",
    }


def test_cli_reports_unknown_chat_without_creating_storage(tmp_path, capsys):
    database = _database(tmp_path)

    result = main(
        [
            "Question",
            "--chat-name",
            "Missing chat",
            "--db",
            str(database),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "no chat named" in captured.err


def test_cli_rejects_missing_database(tmp_path, capsys):
    database = tmp_path / "missing.sqlite"

    result = main(
        [
            "Question",
            "--chat-name",
            "Any chat",
            "--db",
            str(database),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "SQLite database not found" in captured.err
    assert not database.exists()


def test_cli_semantic_namespaces_are_optional(tmp_path, capsys, monkeypatch):
    _database(tmp_path)
    embedding_database = tmp_path / "data" / "embeddings-v2.sqlite"
    embedding_database.touch()
    requests = []

    class FakeSemanticMemory:
        def recall(self, request):
            requests.append(request)
            return SemanticMemoryResult(
                documents=[
                    SemanticDocument(
                        "doc-1",
                        "Volume six",
                        "A relevant semantic result.",
                        score=0.9,
                    )
                ]
            )

    monkeypatch.setattr(
        cli_module,
        "_semantic_memory",
        lambda args, database, resources: FakeSemanticMemory(),
    )

    result = main(
        [
            "Find the relevant notes",
            "--chat-name",
            "Prompt Comparison",
            "--storage-root",
            str(tmp_path),
            "--namespaces",
            "vol_6",
            "vol_7",
            "documents",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "A relevant semantic result." in captured.out
    assert requests[0].namespaces == ["vol_6", "vol_7", "documents"]
    assert requests[0].top_k == 3
    assert requests[0].score_threshold == 0.0
    assert requests[0].max_chars == 1800
    assert "--- Candidate decisions ---" in captured.out
    assert "semantic:doc-1:1" in captured.out
    assert "selected" in captured.out


def test_cli_without_namespaces_does_not_build_semantic_memory(
    tmp_path, capsys, monkeypatch
):
    _database(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("semantic memory must remain disabled")

    monkeypatch.setattr(cli_module, "_semantic_memory", fail_if_called)

    result = main(
        [
            "Episodic only",
            "--chat-name",
            "Prompt Comparison",
            "--storage-root",
            str(tmp_path),
        ]
    )

    assert result == 0


def test_cli_default_policy_limits_and_filters_episodic_events(
    tmp_path, capsys
):
    database = _database(tmp_path)
    with SqliteEpisodicMemory(database) as memory:
        memory.add_events(
            "session-123",
            [
                EpisodicEvent(
                    "system",
                    {"total": 6000},
                    kind="prompt_report",
                ),
                *[
                    EpisodicEvent(
                        "user",
                        f"Recent message {index}",
                        kind="user_message",
                    )
                    for index in range(10)
                ],
            ],
        )

    result = main(
        [
            "Current question",
            "--chat-name",
            "Prompt Comparison",
            "--storage-root",
            str(tmp_path),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    contents = [item["content"] for item in payload["messages"]]
    assert result == 0
    assert not any("6000" in content for content in contents)
    assert payload["metrics"]["episodic_events"]["retrieved_items"] == 6
