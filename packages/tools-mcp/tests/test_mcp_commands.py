from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from harness.cli.mcp_commands import mcp_app

runner = CliRunner()
SERVER = Path(__file__).with_name("stdio_server.py")


def test_list_does_not_resolve_or_print_secrets(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp.servers.remote]\ntransport="streamable-http"\n'
        'url="https://mcp.invalid/mcp?token=do-not-print"\n'
        'bearer_token_env="NOT_PRESENT"\nheaders={X-Test="do-not-print"}\n'
    )
    result = runner.invoke(mcp_app, ["list", "--config", str(config), "--json"])
    assert result.exit_code == 0, result.output
    assert "do-not-print" not in result.output
    assert json.loads(result.output)[0]["name"] == "remote"


def test_check_discovers_actual_stdio_server_then_exits(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[mcp.servers.local]\n"
        f"command={json.dumps(sys.executable)}\nargs=[{json.dumps(str(SERVER))}]\n"
        'include_tools=["echo"]\n'
    )
    result = runner.invoke(
        mcp_app, ["check", "local", "--config", str(config), "--cwd", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["tools"][0]["name"] == "mcp__local__tool__echo"


def test_check_reports_configuration_errors_without_secret_values(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[mcp.servers.bad]\ncommand="python"\nurl="https://do-not-print.invalid"\n')
    result = runner.invoke(mcp_app, ["check", "--config", str(config)])
    assert result.exit_code == 1
    assert "do-not-print" not in result.output
    assert "only valid for streamable-http" in result.output
