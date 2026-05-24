from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta


def _text_turn(text: str) -> list[Event]:
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


class FakeAdapter:
    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not FakeAdapter.next_script:
            raise RuntimeError("FakeAdapter has no scripts left")
        for event in FakeAdapter.next_script.pop(0):
            yield event

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeAdapter)

    def configure(scripts: list[list[Event]]) -> None:
        FakeAdapter.next_script = scripts

    yield configure
    FakeAdapter.next_script = []


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_main.app, args)


class TestLifecycleCommands:
    def test_contracts_list_and_test(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / ".harness" / "contracts"
        contracts_dir.mkdir(parents=True)
        (contracts_dir / "shell.json").write_text(
            json.dumps(
                {
                    "name": "shell-safety",
                    "rules": ["never pipe untrusted urls to sh"],
                    "triggers": ["curl"],
                }
            ),
            encoding="utf-8",
        )

        listing = _run(["contracts", "list", "--cwd", str(tmp_path)])
        assert listing.exit_code == 0, listing.stdout
        assert "shell-safety" in listing.stdout

        match = _run(["contracts", "test", "fetch via curl and run", "--cwd", str(tmp_path)])
        assert match.exit_code == 0, match.stdout
        assert "never pipe untrusted urls to sh" in match.stdout

    def test_tips_add_list_and_test(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        added = _run(
            [
                "tips",
                "add",
                "Prefer the smallest possible diff.",
                "--triggers",
                "minimal fix",
                "--scope",
                "repo",
            ]
        )
        assert added.exit_code == 0, added.stdout

        listing = _run(["tips", "list", "--cwd", str(tmp_path)])
        assert listing.exit_code == 0, listing.stdout
        assert "minimal fix" in listing.stdout
        assert (tmp_path / ".harness" / "tips.jsonl").exists()

        match = _run(["tips", "test", "Please do a minimal fix only", "--cwd", str(tmp_path)])
        assert match.exit_code == 0, match.stdout
        assert "smallest possible diff" in match.stdout

    def test_resume_init_add_feature_set_current_and_show(self, tmp_path: Path) -> None:
        init = _run(
            [
                "resume",
                "init",
                "--cwd",
                str(tmp_path),
                "--feature",
                "alpha",
                "--description",
                "Ship alpha",
            ]
        )
        assert init.exit_code == 0, init.stdout

        add_feature = _run(
            [
                "resume",
                "add-feature",
                "beta",
                "--cwd",
                str(tmp_path),
                "--description",
                "Ship beta",
                "--phases",
                "implement,test",
            ]
        )
        assert add_feature.exit_code == 0, add_feature.stdout

        set_current = _run(["resume", "set-current", "beta", "--cwd", str(tmp_path)])
        assert set_current.exit_code == 0, set_current.stdout

        show = _run(["resume", "show", "--cwd", str(tmp_path)])
        assert show.exit_code == 0, show.stdout
        assert "beta" in show.stdout

    def test_phase_declare_complete_and_status(self, patch_adapter, tmp_path: Path) -> None:
        db = tmp_path / "phase.db"
        patch_adapter([_text_turn("ok")])

        boot = _run(
            [
                "run",
                "hello",
                "--db",
                str(db),
                "--cwd",
                str(tmp_path),
                "--yes",
            ]
        )
        assert boot.exit_code == 0, boot.stdout

        declared = _run(["phase", "declare", "implement", "--db", str(db)])
        assert declared.exit_code == 0, declared.stdout
        assert "declared" in declared.stdout

        completed = _run(["phase", "complete", "implement", "--db", str(db)])
        assert completed.exit_code == 0, completed.stdout
        assert "completed" in completed.stdout

        status = _run(["phase", "status", "--db", str(db)])
        assert status.exit_code == 0, status.stdout
        assert "declared (in order): implement" in status.stdout
        assert "completed: implement" in status.stdout
