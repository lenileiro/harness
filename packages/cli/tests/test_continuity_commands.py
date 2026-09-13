"""Continuity across real CLI entrypoints, using an offline recording adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli
from harness.cli import chat_commands
from harness.core import Capabilities, Done, Message, Session, TextDelta
from harness.storage.sqlite import SQLiteStorage


class RecordingAdapter:
    name = "ollama"
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield TextDelta(text="OK")
        yield Done(final_message=Message(role="assistant", content="OK"))

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(cli, "OllamaAdapter", RecordingAdapter)
    monkeypatch.setattr(
        chat_commands,
        "_classify_chat_turn_policy",
        lambda **kwargs: asyncio.sleep(0, result=chat_commands._GENERAL_TURN_POLICY),
    )
    RecordingAdapter.calls = []
    yield CliRunner()
    RecordingAdapter.calls = []


def test_explicit_chat_skill_survives_restart(offline, tmp_path):
    skill = tmp_path / ".harness/skills/release-check"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: release-check\ndescription: Release checks\n---\nInspect the release manifest.\n"
    )
    args = [
        "chat",
        "--provider",
        "ollama",
        "--model",
        "model",
        "--session",
        "skill-chat",
        "--db",
        str(tmp_path / "db"),
        "--yes",
        "--verify",
        "none",
    ]
    result = offline.invoke(cli.app, args, input="/skill release-check\nhello\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Activated skill" in result.output
    assert any(
        "Active skill release-check" in (message.content or "")
        for message in RecordingAdapter.calls[-1]["messages"]
    )
    result = offline.invoke(cli.app, args, input="continue\n/quit\n")
    assert result.exit_code == 0, result.output
    assert any(
        "Active skill release-check" in (message.content or "")
        for message in RecordingAdapter.calls[-1]["messages"]
    )


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [
        ("/model new-model\nhello\n/quit\n", ["new-model"]),
        ("hello\n/model new-model\nfollow up\n/quit\n", ["old-model", "new-model"]),
        ("hello\n/model new-model\n/quit\n", ["old-model"]),
        (
            "hello\n/model new-model\n/new other\nhello other\n/switch main\nfollow up\n/quit\n",
            ["old-model", "old-model", "new-model"],
        ),
    ],
)
def test_chat_model_selection_is_used_and_survives_restart(offline, tmp_path, stdin, expected):
    db = tmp_path / "chat.db"
    args = ["chat", "--provider", "ollama", "--db", str(db), "--yes", "--verify", "none"]
    result = offline.invoke(
        cli.app, [*args, "--model", "old-model", "--session", "main"], input=stdin
    )
    assert result.exit_code == 0, result.output
    assert [call["model"] for call in RecordingAdapter.calls] == expected

    result = offline.invoke(cli.app, [*args, "--session", "main"], input="continue\n/quit\n")
    assert result.exit_code == 0, result.output
    assert RecordingAdapter.calls[-1]["model"] == "new-model"
    result = offline.invoke(
        cli.app,
        [*args, "--session", "main", "--model", "explicit-model"],
        input="override\n/quit\n",
    )
    assert result.exit_code == 0, result.output
    assert RecordingAdapter.calls[-1]["model"] == "explicit-model"


def test_chat_model_selection_survives_policy_reclassification(offline, monkeypatch):
    policies = iter([chat_commands._GENERAL_TURN_POLICY, chat_commands._REVIEW_TURN_POLICY])
    monkeypatch.setattr(
        chat_commands,
        "_classify_chat_turn_policy",
        lambda **kwargs: asyncio.sleep(0, result=next(policies)),
    )
    result = offline.invoke(
        cli.app,
        ["chat", "--model", "old-model", "--in-memory", "--yes", "--verify", "none"],
        input="hello\n/model new-model\nreview this\n/quit\n",
    )
    assert result.exit_code == 0, result.output
    assert RecordingAdapter.calls[0]["model"] == "old-model"
    assert {call["model"] for call in RecordingAdapter.calls[1:]} == {"new-model"}


def test_memory_defaults_match_initialized_workspace_and_explicit_db_wins(offline, tmp_path):
    result = offline.invoke(cli.app, ["init"])
    assert result.exit_code == 0, result.output
    result = offline.invoke(cli.app, ["memory", "save", "workspace preference"])
    assert result.exit_code == 0, result.output
    entry_id = next(part for part in result.output.split() if part.startswith("mem_"))
    workspace_db = tmp_path / ".harness" / "harness.db"
    result = offline.invoke(cli.app, ["memory", "list", "--db", str(workspace_db)])
    assert "workspace preference" in result.output
    result = offline.invoke(cli.app, ["memory", "search", "workspace"])
    assert "workspace preference" in result.output
    result = offline.invoke(cli.app, ["memory", "list", "--db", str(tmp_path / "other.db")])
    assert "workspace preference" not in result.output

    result = offline.invoke(
        cli.app, ["run", "hello", "--provider", "ollama", "--yes", "--verify", "none"]
    )
    assert result.exit_code == 0, result.output
    assert any(
        "workspace preference" in (m.content or "") for m in RecordingAdapter.calls[-1]["messages"]
    )
    result = offline.invoke(cli.app, ["memory", "rm", entry_id, "--yes"])
    assert result.exit_code == 0, result.output
    result = offline.invoke(cli.app, ["memory", "list", "--db", str(workspace_db)])
    assert "workspace preference" not in result.output


@pytest.mark.parametrize("entrypoint", ["run", "chat", "resume", "fork"])
def test_workspace_guidance_loads_from_selected_cwd_on_every_entrypoint(
    offline, tmp_path, entrypoint
):
    workspace = tmp_path / "workspace"
    state = workspace / ".harness"
    (state / "contracts").mkdir(parents=True)
    (state / "contracts" / "rule.json").write_text(
        json.dumps({"name": "rule", "rules": ["contract marker"]})
    )
    (state / "tips.jsonl").write_text(json.dumps({"text": "tip marker"}) + "\n")
    procedure = state / "procedures" / "guide"
    procedure.mkdir(parents=True)
    (procedure / "procedure.json").write_text(json.dumps({"name": "guide"}))
    (procedure / "PROCEDURE.md").write_text("procedure marker")
    (state / "resume.json").write_text(
        json.dumps(
            {
                "current": "feature",
                "features": [{"name": "feature", "description": "resume marker"}],
            }
        )
    )
    db = tmp_path / "sessions.db"

    async def seed():
        store = SQLiteStorage(path=db)
        try:
            await store.save(
                Session(id="saved", provider="ollama", model="test-model", cwd=workspace)
            )
        finally:
            await store.close()

    asyncio.run(seed())
    if entrypoint in {"resume", "fork"}:
        args = ["sessions", entrypoint, "saved", "continue", "--db", str(db), "--yes"]
    else:
        args = [
            entrypoint,
            *(["continue"] if entrypoint == "run" else []),
            "--cwd",
            str(workspace),
            "--db",
            str(db),
            "--yes",
            "--verify",
            "none",
        ]
    result = offline.invoke(
        cli.app, args, input="continue\n/quit\n" if entrypoint == "chat" else None
    )
    assert result.exit_code == 0, result.output
    rendered = "\n".join(
        m.content or "" for m in RecordingAdapter.calls[-1]["messages"] if m.role == "system"
    )
    for marker in ("contract marker", "tip marker", "procedure marker", "resume marker"):
        assert marker in rendered, (entrypoint, marker, rendered)
