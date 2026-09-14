def test_package_imports():
    import galet_prompt_builder

    assert galet_prompt_builder.__version__ == "0.1.0.dev0"


def test_package_has_no_lucy_imports():
    from pathlib import Path

    root = Path(__file__).parents[1] / "src" / "galet_prompt_builder"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in root.rglob("*.py")
    )
    assert "from src." not in source
    assert "import src." not in source
