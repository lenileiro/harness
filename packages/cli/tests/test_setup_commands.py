from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from harness.cli import setup_commands as commands
from harness.core.skills import SkillLibrary

app = typer.Typer()
app.command("setup")(commands.setup_command)
app.command("doctor")(commands.doctor_command)
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(commands, "inspect_codex_cli_auth", lambda: None)
    monkeypatch.setattr(commands, "inspect_codex_openai_auth", lambda: None)
    monkeypatch.setattr(commands, "codex_cli_available", lambda: False)
    monkeypatch.setattr(commands, "claude_cli_available", lambda: False)
    monkeypatch.setattr(commands, "inspect_claude_cli_auth", lambda binary=None: None)
    monkeypatch.setattr(commands, "default_config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(commands, "default_db_path", lambda: tmp_path / "state" / "sessions.db")
    monkeypatch.setattr(commands, "default_skill_paths", lambda cwd: [cwd / ".harness" / "skills"])


def _setup(config: Path, *extra: str):
    return runner.invoke(
        app,
        ["setup", "--provider", "ollama", "--model", "test-model", "--config", str(config), *extra],
    )


def _doctor(config: Path, cwd: Path, *extra: str):
    return runner.invoke(
        app, ["doctor", "--config", str(config), "--cwd", str(cwd), "--json", *extra]
    )


def test_setup_creates_private_inspectable_config_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-secret-never-persist")
    config = tmp_path / "nested" / "config.toml"
    result = _setup(config)
    assert result.exit_code == 0, result.output
    assert tomllib.loads(config.read_text()) == {
        "default": {"provider": "ollama", "model": "test-model"}
    }
    assert "offline-secret-never-persist" not in config.read_text() + result.output
    if os.name == "posix":
        assert config.stat().st_mode & 0o777 == 0o600


def test_setup_refuses_overwrite_and_force_preserves_unrelated_settings(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = '# keep this comment\n[default]\nprovider="openai"\nmodel="old"\n[approval]\nshell="deny"\n[provider.openai]\ntimeout=123\n'
    config.write_text(original)
    result = _setup(config)
    assert result.exit_code == 1 and "--force" in result.output
    assert config.read_text() == original
    result = _setup(config, "--force")
    assert result.exit_code == 0, result.output
    data = tomllib.loads(config.read_text())
    assert data["approval"] == {"shell": "deny"}
    assert data["provider"]["openai"] == {"timeout": 123}
    assert data["default"] == {"provider": "ollama", "model": "test-model"}
    assert "# keep this comment" in config.read_text()


def test_setup_failed_validation_leaves_original_untouched(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = '[default]\nprovider="ollama"\n[approval]\nshell="offline-secret-invalid-value"\n'
    config.write_text(original)
    result = _setup(config, "--force")
    assert result.exit_code == 1
    assert "offline-secret-invalid-value" not in result.output
    assert config.read_text() == original
    assert list(tmp_path.glob(".harness-config-*")) == []


def test_setup_supports_inline_default_and_unicode_without_losing_tables(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('default={provider="ollama", model="old"}\nextra={label="🙂"}\n')
    result = _setup(config, "--force")
    assert result.exit_code == 0, result.output
    assert tomllib.loads(config.read_text())["extra"] == {"label": "🙂"}


def test_doctor_is_offline_and_never_executes_configured_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[default]\nprovider="ollama"\n[mcp.servers.offline]\ncommand={json.dumps(sys.executable)}\nargs=["-c", "raise RuntimeError()"]\n'
    )

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("MCP connections require explicit --connect")

    monkeypatch.setattr(commands, "MCPToolset", forbidden)
    result = _doctor(config, tmp_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["mcp:offline"]["status"] == "ok"
    assert "not connected" in checks["mcp:offline"]["detail"]
    assert checks["provider:ollama"]["status"] == "warning"
    assert "not been probed" in checks["provider:ollama"]["detail"]
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / ".harness").exists()


def test_doctor_reports_native_anthropic_credentials_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="anthropic"\n')
    missing = _doctor(config, tmp_path)
    assert missing.exit_code == 1 and "ANTHROPIC_API_KEY is missing" in missing.output
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-secret-never-print")
    configured = _doctor(config, tmp_path)
    assert configured.exit_code == 0, configured.output
    assert "credential present; not verified" in configured.output
    assert "offline-secret-never-print" not in configured.output


def test_doctor_reports_claude_cli_login_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="claude"\n')
    missing = _doctor(config, tmp_path)
    assert missing.exit_code == 1 and "Claude Code CLI executable is missing" in missing.output

    monkeypatch.setattr(commands, "claude_cli_available", lambda: True)
    monkeypatch.setattr(commands, "inspect_claude_cli_auth", lambda binary=None: None)
    logged_out = _doctor(config, tmp_path)
    assert logged_out.exit_code == 1 and "claude auth login" in logged_out.output

    monkeypatch.setattr(
        commands,
        "inspect_claude_cli_auth",
        lambda binary=None: {"logged_in": True, "auth_method": "claude.ai"},
    )
    ready = _doctor(config, tmp_path)
    assert ready.exit_code == 0, ready.output
    assert "claude.ai login present; not verified" in ready.output


def test_doctor_rejects_empty_codex_auth_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        commands,
        "inspect_codex_cli_auth",
        lambda: {"auth_mode": "unknown", "has_openai_api_key": False, "has_access_token": False},
    )
    monkeypatch.setattr(commands, "codex_cli_available", lambda: True)
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="codex"\n')
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1 and "saved Codex login is missing" in result.output


def test_doctor_reports_config_and_mcp_credential_failures_without_values(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp.servers.remote]\ntransport="streamable-http"\nurl="https://offline.invalid/mcp"\nbearer_token_env="MISSING_TEST_DOCTOR_TOKEN"\n'
    )
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1
    assert "required environment reference" in result.output
    config.write_text('[approval]\nshell="offline-secret-invalid"\n')
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1 and "offline-secret-invalid" not in result.output


def test_doctor_validates_skills_without_rendering_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="ollama"\n')
    skill = tmp_path / ".harness" / "skills" / "test-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: wrong-name\ndescription: test\n---\nbody-secret-must-not-render\n"
    )

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("doctor must not render skill bodies")

    monkeypatch.setattr(SkillLibrary, "render_context", forbidden)
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1 and "1 invalid" in result.output
    assert "body-secret-must-not-render" not in result.output


def test_doctor_detects_unwritable_database_target(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="ollama"\n')
    directory = tmp_path / "database-is-a-directory"
    directory.mkdir()
    result = _doctor(config, tmp_path, "--db", str(directory))
    assert result.exit_code == 1 and "database path or its parent is not writable" in result.output


def test_doctor_connect_only_runs_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp.servers.remote]\ntransport="streamable-http"\nurl="https://offline.invalid/mcp"\n'
    )
    connected = []

    class FakeToolset:
        tools = (object(),)

        def __init__(self, servers: Any, **kwargs: Any):
            connected.extend(server.name for server in servers)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any):
            return None

    monkeypatch.setattr(commands, "MCPToolset", FakeToolset)
    result = _doctor(config, tmp_path, "--connect")
    assert result.exit_code == 0, result.output
    assert connected == ["remote"]
    assert "1 tools discovered" in result.output


def test_doctor_reports_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="ollama"\n')
    monkeypatch.setattr(commands, "_module_available", lambda name: name != "mcp")
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1
    assert any(
        check["name"] == "dependency:mcp" and check["status"] == "error"
        for check in json.loads(result.output)["checks"]
    )


def test_custom_provider_setup_and_doctor(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        '[provider.private]\ndriver="openai-compatible"\nbase_url="https://example.invalid/v1"\napi_key_env="PRIVATE_MODEL_TEST_KEY"\n'
    )
    result = runner.invoke(
        app,
        [
            "setup",
            "--provider",
            "private",
            "--model",
            "example",
            "--config",
            str(config),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    result = _doctor(config, tmp_path)
    assert result.exit_code == 1
    monkeypatch.setenv("PRIVATE_MODEL_TEST_KEY", "private-value")
    result = _doctor(config, tmp_path)
    assert result.exit_code == 0, result.output
    assert "private-value" not in result.output
    checks = {row["name"]: row for row in json.loads(result.output)["checks"]}
    assert checks["provider:private"]["status"] == "ok"


def test_doctor_checks_managed_integration_references_without_opening_clients(
    tmp_path, monkeypatch
):
    import httpx

    config = tmp_path / "config.toml"
    config.write_text("""
[default]
provider="ollama"
[computer]
enabled=true
[honcho]
enabled=true
api_key_env="DOCTOR_HONCHO"
[homeassistant]
enabled=true
base_url="https://ha.example"
token_env="DOCTOR_HA"
[a2a]
enabled=true
[a2a.peers.peer]
url="https://peer.example/a2a"
token_env="DOCTOR_PEER"
[portal]
enabled=true
provider="missing-account"
routes=["web"]
""")

    def forbidden(*args, **kwargs):
        pytest.fail("Offline doctor must not open network clients")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(commands, "_module_available", lambda name: name != "pyautogui")
    for variable in ["DOCTOR_HONCHO", "DOCTOR_HA", "DOCTOR_PEER"]:
        monkeypatch.delenv(variable, raising=False)
    first = _doctor(config, tmp_path)
    checks = {item["name"]: item for item in json.loads(first.output)["checks"]}
    assert first.exit_code == 1
    assert all(
        checks[name]["status"] == "error"
        for name in ["computer", "honcho", "homeassistant", "a2a:peer", "portal"]
    )
    secret = "doctor-private-secret-that-must-not-appear"
    for variable in ["DOCTOR_HONCHO", "DOCTOR_HA", "DOCTOR_PEER"]:
        monkeypatch.setenv(variable, secret)
    second = _doctor(config, tmp_path)
    checks = {item["name"]: item for item in json.loads(second.output)["checks"]}
    assert all(checks[name]["status"] == "ok" for name in ["honcho", "homeassistant", "a2a:peer"])
    assert checks["computer"]["status"] == "error" and checks["portal"]["status"] == "error"
    assert secret not in second.output
    assert not (tmp_path / ".harness").exists()
