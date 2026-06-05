"""End-to-end CLI integration tests for `harness run`.

The Ollama adapter is swapped for a deterministic in-process fake so these
tests run without any external daemon, while exercising the real Agent +
ToolRegistry + InMemoryStorage + Rich-rendering wiring.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import run_commands as run_mod
from harness.cli.config import HarnessConfig
from harness.core import (
    Capabilities,
    DomainProfile,
    Done,
    ErrorEvent,
    Event,
    Message,
    TextDelta,
    Tip,
    ToolCall,
    ToolCallEvent,
)

# ---------------------------------------------------------------------------
# Fake adapter
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Drop-in replacement for OllamaAdapter that replays a fixed event list.

    Scripts pop from the class-level queue on every stream() call so a test
    can drive multiple CLI invocations from a single pre-loaded list.
    """

    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not FakeAdapter.next_script:
            raise RuntimeError("FakeAdapter has no scripts left for this stream call")
        for event in FakeAdapter.next_script.pop(0):
            yield event

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


class OpenRouterAdapterProbe:
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).captured = kwargs


class CodexAdapterProbe:
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).captured = kwargs


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    """Swap the real OllamaAdapter for FakeAdapter and let the test pre-load scripts."""
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeAdapter)

    def configure(scripts: list[list[Event]]) -> None:
        FakeAdapter.next_script = scripts

    yield configure


def test_cli_openrouter_adapter_respects_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main, "OpenRouterAdapter", OpenRouterAdapterProbe)
    cfg = HarnessConfig(provider_settings={"openrouter": {"timeout": 23.0}})

    cli_main._build_adapter("openrouter", base_url=None, config=cfg)

    assert OpenRouterAdapterProbe.captured["timeout"] == 23.0
    FakeAdapter.next_script = []


def test_cli_codex_adapter_uses_model_timeout_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main, "CodexAdapter", CodexAdapterProbe)
    monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "900")
    monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "180")
    cfg = HarnessConfig(provider_settings={"codex": {"timeout": 600.0, "idle_timeout": 120.0}})

    cli_main._build_adapter("codex", base_url=None, config=cfg)

    assert CodexAdapterProbe.captured["timeout"] == 900.0
    assert CodexAdapterProbe.captured["idle_timeout"] == 180.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_text_response_streams_to_stdout(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter(
            [
                [
                    TextDelta(text="hello "),
                    TextDelta(text="world"),
                    Done(final_message=Message(role="assistant", content="hello world")),
                ]
            ]
        )
        runner = CliRunner()
        result = runner.invoke(
            cli_main.app,
            [
                "run",
                "say hi",
                "--cwd",
                str(tmp_path),
                "--model",
                "test-model",
                "--in-memory",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "hello world" in result.stdout

    def test_tool_call_then_final_answer(self, patch_adapter, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("the answer is 42", encoding="utf-8")

        first_call = ToolCall(id="c1", name="read_file", arguments={"path": "note.txt"})
        patch_adapter(
            [
                [
                    ToolCallEvent(call=first_call),
                    Done(
                        final_message=Message(
                            role="assistant",
                            content=None,
                            tool_calls=[first_call],
                        )
                    ),
                ],
                [
                    TextDelta(text="the answer is 42"),
                    Done(final_message=Message(role="assistant", content="the answer is 42")),
                ],
            ]
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main.app,
            ["run", "what is in note.txt?", "--cwd", str(tmp_path), "--in-memory"],
        )
        assert result.exit_code == 0, result.stdout
        # Tool name shows up in the rendered tool-call line.
        assert "read_file" in result.stdout
        # Final answer streams to stdout.
        assert "the answer is 42" in result.stdout

    def test_missing_cwd_exits_2(self, tmp_path: Path) -> None:
        ghost = tmp_path / "does-not-exist"
        runner = CliRunner()
        result = runner.invoke(cli_main.app, ["run", "x", "--cwd", str(ghost), "--in-memory"])
        assert result.exit_code == 2

    def test_adaptive_profile_announces_strategy(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter(
            [
                [
                    TextDelta(text="done"),
                    Done(final_message=Message(role="assistant", content="done")),
                ]
            ]
        )
        runner = CliRunner()
        result = runner.invoke(
            cli_main.app,
            [
                "run",
                "Fix the bug with a minimal fix only.",
                "--cwd",
                str(tmp_path),
                "--model",
                "test-model",
                "--in-memory",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "adaptive strategy" in result.stdout

    def test_run_once_uses_plugin_domain_and_experience_providers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class DemoDomainProvider:
            def profiles(self) -> list[DomainProfile]:
                return [
                    DomainProfile(
                        name="docs-review",
                        description="Plugin profile",
                        allowed_tools=("read_file",),
                        system_prompt="PLUGIN PROMPT",
                    )
                ]

        class DemoExperienceProvider:
            def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
                return [Tip(text="plugin guidance", triggers=("pytest",), weight=4.0)]

        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_domain_profile_providers",
            lambda cwd, *, config: [DemoDomainProvider()],
        )
        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_experience_providers",
            lambda cwd, *, config: [DemoExperienceProvider()],
        )

        captured: dict[str, object] = {}

        class FakeAgent:
            async def run(self, request: object) -> AsyncIterator[Event]:
                yield Done(final_message=Message(role="assistant", content="reviewed"))

        async def fake_resolve_task_attachment(
            storage: object, task_ref: object, session_id: object
        ):
            return None, None

        def fake_build_tools(
            tool_cwd: Path, *, config: HarnessConfig, include: set[str] | None = None
        ):
            captured["include"] = include
            return include

        def fake_build_agent(**kwargs: object) -> FakeAgent:
            captured["system_prompt"] = kwargs["system_prompt"]
            captured["verify_command"] = kwargs["verify_command"]
            captured["tips"] = [
                tip.text
                for tip in kwargs["tips_provider"].query("pytest", top_k=5)  # type: ignore[union-attr]
            ]
            build_tools = cast(Callable[[Path], object], kwargs["build_tools"])
            build_tools(tmp_path)
            return FakeAgent()

        stream = StringIO()
        console = Console(file=stream, force_terminal=False, color_system=None)
        final = asyncio.run(
            run_mod.run_once(
                prompt="review the docs patch",
                model="test-model",
                chain=["ollama"],
                base_url=None,
                cwd=tmp_path,
                max_steps=4,
                max_output_tokens=None,
                session_id=None,
                task_ref=None,
                db=None,
                in_memory=True,
                yes=True,
                inbox=False,
                verify="none",
                verify_command="echo ok",
                critic=None,
                require_tools=False,
                goal=False,
                max_context_tokens=None,
                predict=False,
                auto_compact=False,
                max_repair=1,
                profile="minimal",
                domain="docs-review",
                phases=None,
                loop_detect=False,
                contracts=False,
                tips=True,
                config=HarnessConfig(),
                build_storage=lambda **kwargs: object(),
                resolve_task_attachment=fake_resolve_task_attachment,
                resolve_runtime_strategy=lambda **kwargs: type(
                    "Strategy",
                    (),
                    {"structural_profile": "minimal", "critic_mode": "none", "rationale": "x"},
                )(),
                build_verifier=lambda *args, **kwargs: None,
                build_critic=lambda *args, **kwargs: None,
                build_adapter=lambda *args, **kwargs: None,
                build_tools=fake_build_tools,
                build_agent=fake_build_agent,
                print_defense_ledger=lambda *args, **kwargs: asyncio.sleep(0),
                render=lambda event: None,
                default_system_prompt="DEFAULT PROMPT",
                console=console,
            )
        )

        assert final == "reviewed"
        assert captured["system_prompt"] == "PLUGIN PROMPT"
        assert captured["verify_command"] == "echo ok"
        assert captured["include"] == {"read_file"}
        assert captured["tips"] == ["plugin guidance"]

    def test_run_once_can_disable_workspace_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_domain_profile_providers",
            lambda cwd, *, config: [],
        )
        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_experience_providers",
            lambda cwd, *, config: [],
        )
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        (harness_dir / "resume.json").write_text(
            json.dumps(
                {
                    "current": "checkout-migration",
                    "features": [
                        {
                            "name": "checkout-migration",
                            "description": "Migrate checkout.",
                            "status": "in_progress",
                        }
                    ],
                    "notes": [],
                }
            ),
            encoding="utf-8",
        )
        contracts_dir = harness_dir / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "checkout.json").write_text(
            json.dumps({"name": "checkout", "rules": ["Stay on checkout."], "triggers": []}),
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        class FakeAgent:
            async def run(self, request: object) -> AsyncIterator[Event]:
                yield Done(final_message=Message(role="assistant", content="answered"))

        async def fake_resolve_task_attachment(
            storage: object, task_ref: object, session_id: object
        ):
            return None, None

        def fake_build_agent(**kwargs: object) -> FakeAgent:
            captured.update(kwargs)
            return FakeAgent()

        final = asyncio.run(
            run_mod.run_once(
                prompt="What is the weather in Tokyo?",
                model="test-model",
                chain=["ollama"],
                base_url=None,
                cwd=tmp_path,
                max_steps=4,
                max_output_tokens=None,
                session_id=None,
                task_ref=None,
                db=None,
                in_memory=True,
                yes=True,
                inbox=False,
                verify="none",
                critic=None,
                require_tools=False,
                goal=False,
                max_context_tokens=None,
                predict=False,
                auto_compact=False,
                max_repair=1,
                profile="minimal",
                domain="coding",
                phases=None,
                loop_detect=True,
                contracts=True,
                tips=True,
                include_workspace_context=False,
                config=HarnessConfig(),
                build_storage=lambda **kwargs: object(),
                resolve_task_attachment=fake_resolve_task_attachment,
                resolve_runtime_strategy=lambda **kwargs: type(
                    "Strategy",
                    (),
                    {"structural_profile": "minimal", "critic_mode": "none", "rationale": "x"},
                )(),
                build_verifier=lambda *args, **kwargs: None,
                build_critic=lambda *args, **kwargs: None,
                build_adapter=lambda *args, **kwargs: None,
                build_tools=lambda *args, **kwargs: object(),
                build_agent=fake_build_agent,
                print_defense_ledger=lambda *args, **kwargs: asyncio.sleep(0),
                render=lambda event: None,
                default_system_prompt="DEFAULT PROMPT",
                console=Console(file=StringIO(), force_terminal=False, color_system=None),
            )
        )

        assert final == "answered"
        assert captured["memory_store"] is None
        assert captured["contracts"] is None
        assert captured["tips_provider"] is None
        assert captured["resume"] is None

    def test_run_once_uses_loop_detector_for_bare_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_domain_profile_providers",
            lambda cwd, *, config: [],
        )

        captured: dict[str, object] = {}

        class FakeAgent:
            async def run(self, request: object) -> AsyncIterator[Event]:
                yield Done(final_message=Message(role="assistant", content="answered"))

        async def fake_resolve_task_attachment(
            storage: object, task_ref: object, session_id: object
        ):
            return None, None

        def fake_build_agent(**kwargs: object) -> FakeAgent:
            captured.update(kwargs)
            return FakeAgent()

        final = asyncio.run(
            run_mod.run_once(
                prompt="write the script",
                model="test-model",
                chain=["ollama"],
                base_url=None,
                cwd=tmp_path,
                max_steps=4,
                max_output_tokens=None,
                session_id=None,
                task_ref=None,
                db=None,
                in_memory=True,
                yes=True,
                inbox=False,
                verify="none",
                critic=None,
                require_tools=False,
                goal=False,
                max_context_tokens=None,
                predict=False,
                auto_compact=False,
                max_repair=1,
                profile="bare",
                domain="coding",
                phases=None,
                loop_detect=True,
                contracts=False,
                tips=False,
                include_workspace_context=False,
                config=HarnessConfig(),
                build_storage=lambda **kwargs: object(),
                resolve_task_attachment=fake_resolve_task_attachment,
                resolve_runtime_strategy=lambda **kwargs: type(
                    "Strategy",
                    (),
                    {"structural_profile": "bare", "critic_mode": "none", "rationale": "x"},
                )(),
                build_verifier=lambda *args, **kwargs: None,
                build_critic=lambda *args, **kwargs: None,
                build_adapter=lambda *args, **kwargs: None,
                build_tools=lambda *args, **kwargs: object(),
                build_agent=fake_build_agent,
                print_defense_ledger=lambda *args, **kwargs: asyncio.sleep(0),
                render=lambda event: None,
                default_system_prompt="DEFAULT PROMPT",
                console=Console(file=StringIO(), force_terminal=False, color_system=None),
            )
        )

        assert final == "answered"
        assert captured["loop_detector"].__class__.__name__ == "LoopDetector"

    def test_run_once_does_not_return_streamed_partials_after_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "harness.cli.plugins.load_cli_domain_profile_providers",
            lambda cwd, *, config: [],
        )

        class FakeAgent:
            async def run(self, request: object) -> AsyncIterator[Event]:
                yield TextDelta(text="I cannot fetch weather.")
                yield ErrorEvent(
                    error="exceeded max_steps=3 without final answer",
                    kind="internal",
                    recoverable=False,
                )

        async def fake_resolve_task_attachment(
            storage: object, task_ref: object, session_id: object
        ):
            return None, None

        with pytest.raises(typer.Exit):
            asyncio.run(
                run_mod.run_once(
                    prompt="What is the weather in Tokyo?",
                    model="test-model",
                    chain=["ollama"],
                    base_url=None,
                    cwd=tmp_path,
                    max_steps=3,
                    max_output_tokens=None,
                    session_id=None,
                    task_ref=None,
                    db=None,
                    in_memory=True,
                    yes=True,
                    inbox=False,
                    verify="none",
                    critic=None,
                    require_tools=True,
                    goal=False,
                    max_context_tokens=None,
                    predict=False,
                    auto_compact=False,
                    max_repair=1,
                    profile="minimal",
                    domain="coding",
                    phases=None,
                    loop_detect=True,
                    contracts=False,
                    tips=False,
                    config=HarnessConfig(),
                    build_storage=lambda **kwargs: object(),
                    resolve_task_attachment=fake_resolve_task_attachment,
                    resolve_runtime_strategy=lambda **kwargs: type(
                        "Strategy",
                        (),
                        {"structural_profile": "minimal", "critic_mode": "none", "rationale": "x"},
                    )(),
                    build_verifier=lambda *args, **kwargs: None,
                    build_critic=lambda *args, **kwargs: None,
                    build_adapter=lambda *args, **kwargs: None,
                    build_tools=lambda *args, **kwargs: object(),
                    build_agent=lambda **kwargs: FakeAgent(),
                    print_defense_ledger=lambda *args, **kwargs: asyncio.sleep(0),
                    render=lambda event: None,
                    default_system_prompt="DEFAULT PROMPT",
                    console=Console(file=StringIO(), force_terminal=False, color_system=None),
                )
            )
