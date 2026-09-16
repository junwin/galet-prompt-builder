import json

from galet_memory import EpisodicEvent, SqliteEpisodicMemory

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
