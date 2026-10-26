"""Unit/integration fixtures must not select a developer's application stores."""
import pytest


@pytest.fixture(autouse=True)
def isolate_configuration_environment(monkeypatch):
    monkeypatch.delenv('GALET_PROMPT_BUILDER_CONFIG_PATH', raising=False)
    monkeypatch.delenv('GALET_MEMORY_CONFIG_PATH', raising=False)
