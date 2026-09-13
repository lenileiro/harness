import pytest
import typer
from typer.testing import CliRunner

from harness.cli.serve_commands import make_serve_command


def application(captured):
    app = typer.Typer()

    def factory(workspace, config, adapter, model):
        captured["builder"] = (workspace, config, adapter, model)

        def unused(context):
            raise AssertionError("No live agents in CLI assembly test")

        return unused

    app.command()(make_serve_command(factory))
    return app


def test_serve_default_loopback_and_explicit_token_reference(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("TEST_SERVER_AUTH", "z" * 40)
    monkeypatch.setattr(
        "harness.cli.serve_commands.uvicorn.run",
        lambda app, **kwargs: captured.update({"app": app, **kwargs}),
    )
    result = CliRunner().invoke(
        application(captured),
        [
            "--workspace",
            str(tmp_path),
            "--token-env",
            "alice:TEST_SERVER_AUTH",
            "--mcp",
            "--a2a-url",
            "http://127.0.0.1:8765/a2a",
            "--expose-tool",
            "record",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1" and captured["port"] == 8765
    assert captured["builder"][0] == tmp_path
    assert captured["access_log"] is False
    assert "/a2a" in {route.path for route in captured["app"].routes}
    assert "z" * 40 not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--host", "0.0.0.0"],
        ["--host", "0.0.0.0", "--allow-remote"],
        ["--allowed-host", "*"],
        ["--allowed-origin", "*"],
        ["--a2a-url", "https://credential:secret@server.invalid/a2a"],
    ],
)
def test_serve_rejects_unsafe_bind_or_browser_configuration(tmp_path, monkeypatch, arguments):
    captured = {}
    monkeypatch.setenv("TEST_SERVER_AUTH", "z" * 40)
    monkeypatch.setattr(
        "harness.cli.serve_commands.uvicorn.run",
        lambda *args, **kwargs: captured.update(started=True),
    )
    result = CliRunner().invoke(
        application(captured),
        ["--workspace", str(tmp_path), "--token-env", "alice:TEST_SERVER_AUTH", *arguments],
    )
    assert result.exit_code != 0
    assert "started" not in captured


def test_serve_missing_auth_fails_before_building_agent(tmp_path):
    captured = {}
    result = CliRunner().invoke(application(captured), ["--workspace", str(tmp_path)])
    assert result.exit_code != 0
    assert "builder" not in captured


def test_serve_callbacks_load_only_explicit_grants_and_environment_references(
    tmp_path, monkeypatch
):
    import json

    captured = {}
    monkeypatch.setenv("TEST_SERVER_AUTH", "z" * 40)
    monkeypatch.setenv("TEST_CALLBACK_SECRET", "s" * 40)
    monkeypatch.setattr(
        "harness.cli.serve_commands.uvicorn.run",
        lambda app, **kwargs: captured.update(app=app),
    )
    grants = tmp_path / "callbacks.json"
    grants.write_text(
        json.dumps(
            [
                {
                    "owner": "alice",
                    "url": "https://callback.example/a2a",
                    "secret_env": "TEST_CALLBACK_SECRET",
                }
            ]
        )
    )
    args = [
        "--workspace",
        str(tmp_path),
        "--token-env",
        "alice:TEST_SERVER_AUTH",
        "--a2a-url",
        "http://127.0.0.1:8765/a2a",
        "--a2a-callbacks",
        str(grants),
    ]
    result = CliRunner().invoke(application(captured), args)
    assert result.exit_code == 0, result.output
    assert "/v1/a2a/callback-deliveries" in {route.path for route in captured["app"].routes}
    assert "s" * 40 not in result.output
    captured.clear()
    grants.write_text(
        json.dumps(
            [
                {
                    "owner": "alice",
                    "url": "https://callback.example/a2a",
                    "secret": "DO-NOT-PRINT-INLINE-CREDENTIAL",
                }
            ]
        )
    )
    result = CliRunner().invoke(application(captured), args)
    assert result.exit_code != 0 and "app" not in captured
    assert "DO-NOT-PRINT-INLINE-CREDENTIAL" not in result.output
