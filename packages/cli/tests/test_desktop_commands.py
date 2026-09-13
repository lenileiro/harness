from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from harness.cli.desktop_commands import desktop_command


class Loaded:
    def __init__(self):
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


def app():
    result = typer.Typer()
    result.command("desktop")(desktop_command)
    return result


def test_desktop_uses_private_window_and_origin_guarded_environment_token(monkeypatch):
    captured = {}
    callbacks = Loaded()
    scripts = []
    window = SimpleNamespace(events=SimpleNamespace(loaded=callbacks), run_js=scripts.append)

    def create_window(title, url, **kwargs):
        captured.update(title=title, url=url, **kwargs)
        return window

    def start(**kwargs):
        captured.update(kwargs)
        for callback in callbacks.callbacks:
            callback()

    monkeypatch.setenv("DESKTOP_TEST_TOKEN", "dummy-desktop-token-" + "x" * 40)
    monkeypatch.setattr(
        "harness.cli.desktop_commands.importlib.import_module",
        lambda _: SimpleNamespace(create_window=create_window, start=start, settings={}),
    )
    result = CliRunner().invoke(app(), ["--token-env", "DESKTOP_TEST_TOKEN"])
    assert result.exit_code == 0, result.output
    assert captured["url"] == "http://127.0.0.1:8765/"
    assert captured["private_mode"] is True and captured["debug"] is False
    assert "window.location.origin ===" in scripts[0]
    assert "dummy-desktop-token" in scripts[0] and "dummy-desktop-token" not in result.output


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "file:///tmp/app",
        "https://user:pass@example.com",
        "https://example.com/?token=bad",
        "https://example.com/#token=bad",
    ],
)
def test_desktop_rejects_unsafe_urls_without_loading_webview(url, monkeypatch):
    monkeypatch.setattr(
        "harness.cli.desktop_commands.importlib.import_module",
        lambda _: pytest.fail("must validate before loading native window"),
    )
    result = CliRunner().invoke(app(), ["--url", url])
    assert result.exit_code != 0


def test_desktop_missing_extra_reports_install_instruction(monkeypatch):
    def missing(_):
        raise ImportError

    monkeypatch.setattr("harness.cli.desktop_commands.importlib.import_module", missing)
    result = CliRunner().invoke(app(), [])
    assert result.exit_code != 0
    assert "desktop" in result.output and "uv sync" in result.output
