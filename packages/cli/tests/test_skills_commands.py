import json

from typer.testing import CliRunner

from harness.cli.__main__ import app


def test_create_inspect_validate_and_configured_discovery(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    runner = CliRunner()
    created = runner.invoke(
        app, ["skills", "create", "check-release", "--description", "Check release metadata"]
    )
    assert created.exit_code == 0, created.output
    path = tmp_path / ".harness/skills/check-release/SKILL.md"
    assert path.exists()
    shown = runner.invoke(app, ["skills", "show", "check-release"])
    assert shown.exit_code == 0 and "Check release metadata" in shown.output
    assert runner.invoke(app, ["skills", "validate"]).exit_code == 0
    original = path.read_text()
    assert (
        runner.invoke(
            app, ["skills", "create", "check-release", "--description", "overwrite"]
        ).exit_code
        != 0
    )
    assert path.read_text() == original
    config = tmp_path / "custom.toml"
    config.write_text("[skills]\nenabled=false\n")
    listed = runner.invoke(app, ["skills", "list", "--config", str(config)])
    assert listed.exit_code == 0 and json.loads(listed.output)["skills"] == []
    path.write_text("not valid frontmatter")
    assert runner.invoke(app, ["skills", "validate"]).exit_code == 1


def test_invalid_mcp_config_is_rejected_without_launch(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp.servers.example]\ntransport="streamable-http"\nurl="file:///etc/passwd"\n'
    )
    result = CliRunner().invoke(app, ["skills", "list", "--config", str(config)])
    assert result.exit_code == 2
    assert "invalid MCP configuration" in result.output
