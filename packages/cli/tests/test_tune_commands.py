from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta
from harness.core.verifier_tuner import TunablePrompt


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


def _seed_prompt(path: Path, key: str) -> None:
    prompt = TunablePrompt(key=key)
    prompt.add_version("seed prompt", rationale="initial")
    prompt.add_version("revised prompt", rationale="tuned")
    prompt.save(path)


class TestTuneCommands:
    def test_list_show_and_rollback(self, tmp_path: Path) -> None:
        tuned_dir = tmp_path / ".harness" / "tuned-prompts"
        tuned_dir.mkdir(parents=True)
        path = tuned_dir / "minimal_fix_verifier.json"
        _seed_prompt(path, "minimal_fix_verifier")

        listing = _run(["tune", "list", "--cwd", str(tmp_path)])
        assert listing.exit_code == 0, listing.stdout
        assert "minimal_fix_verifier" in listing.stdout

        show = _run(["tune", "show", "minimal_fix_verifier", "--cwd", str(tmp_path)])
        assert show.exit_code == 0, show.stdout
        assert "revised prompt" in show.stdout

        rollback = _run(["tune", "rollback", "minimal_fix_verifier", "--cwd", str(tmp_path)])
        assert rollback.exit_code == 0, rollback.stdout
        reloaded = TunablePrompt.load(path)
        assert reloaded is not None
        assert reloaded.current is not None
        assert reloaded.current.text == "seed prompt"

    def test_propose_dry_run(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        current = tmp_path / "current.txt"
        current.write_text("Block if diff > 50 lines.", encoding="utf-8")
        pairs = tmp_path / "pairs.json"
        pairs.write_text(
            json.dumps(
                [
                    {
                        "fixture": "f01",
                        "defended_excerpt": "agent ran tests first",
                        "defended_outcome": "PASS overall=5/5",
                        "bare_excerpt": "agent jumped to edit",
                        "bare_outcome": "FAIL scope=1/5",
                        "differing_dimension": "scope",
                    }
                ]
            ),
            encoding="utf-8",
        )
        patch_adapter(
            [
                [
                    TextDelta(
                        text=json.dumps(
                            {
                                "new_prompt": "Prefer the minimum diff that fixes the issue.",
                                "rationale": "tighten scope discipline",
                            }
                        )
                    ),
                    Done(
                        final_message=Message(
                            role="assistant",
                            content=(
                                '{"new_prompt":"Prefer the minimum diff that fixes the issue.",'
                                '"rationale":"tighten scope discipline"}'
                            ),
                        )
                    ),
                ]
            ]
        )

        result = _run(
            [
                "tune",
                "propose",
                "minimal_fix_verifier",
                "--current",
                str(current),
                "--pairs",
                str(pairs),
                "--dry-run",
            ]
        )
        assert result.exit_code == 0, result.stdout
        assert "Prefer the minimum diff that fixes the issue." in result.stdout
        assert "--dry-run: not saving." in result.stdout
        assert not (tmp_path / ".harness" / "tuned-prompts").exists()
