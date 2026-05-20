from typer.testing import CliRunner

from harness.cli import __version__
from harness.cli.__main__ import app


def test_version_is_string() -> None:
    assert isinstance(__version__, str)


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
