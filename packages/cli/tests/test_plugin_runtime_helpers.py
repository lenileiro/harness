from __future__ import annotations

from pathlib import Path

import pytest
import typer

from harness.cli.config import HarnessConfig
from harness.cli.runtime_helpers import build_critic, build_verifier
from harness.core import Capabilities


class FakeAdapter:
    name = "fake"

    def stream(self, **kwargs):
        raise NotImplementedError

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        return None


def test_build_verifier_can_load_plugin_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "plugin_verifier.py").write_text(
        """
from harness.core.schemas import VerificationResult


class DemoVerifier:
    async def verify(self, *, session, activity):
        return VerificationResult(can_finish=True, confidence=1.0, reason="ok")


class DemoProvider:
    def verifiers(self):
        return [DemoVerifier()]
""",
        encoding="utf-8",
    )
    plugin_dir = tmp_path / ".harness" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "verify.toml").write_text(
        'name = "demo-verify"\nkind = "verifier"\nprovider = "plugin_verifier:DemoProvider"\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    verifier = build_verifier(
        "plugin:demo-verify",
        chain=["ollama"],
        model="m",
        config=HarnessConfig(),
        build_adapter=lambda *args, **kwargs: FakeAdapter(),
        cwd=tmp_path,
    )
    assert verifier is not None


def test_build_critic_can_load_plugin_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "plugin_critic.py").write_text(
        """
class DemoCritic:
    async def critique(self, *, session, verification_result, activity):
        return "re-check the premise"


class DemoProvider:
    def critics(self):
        return [DemoCritic()]
""",
        encoding="utf-8",
    )
    plugin_dir = tmp_path / ".harness" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "critic.toml").write_text(
        'name = "demo-critic"\nkind = "critic"\nprovider = "plugin_critic:DemoProvider"\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)

    critic = build_critic(
        "plugin:demo-critic",
        chain=["ollama"],
        model="m",
        config=HarnessConfig(),
        build_adapter=lambda *args, **kwargs: FakeAdapter(),
    )
    assert critic is not None


def test_build_verifier_rejects_unknown_plugin(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        build_verifier(
            "plugin:nope",
            chain=["ollama"],
            model="m",
            config=HarnessConfig(),
            build_adapter=lambda *args, **kwargs: FakeAdapter(),
            cwd=tmp_path,
        )
