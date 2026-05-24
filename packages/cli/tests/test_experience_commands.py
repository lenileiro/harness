from __future__ import annotations

from pathlib import Path

from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, args)


def test_experience_procedures_add_and_list(tmp_path: Path) -> None:
    added = _run(
        [
            "experience",
            "procedures",
            "add",
            "--name",
            "Scope discipline",
            "--body",
            "Keep the patch in the named formatter only.",
            "--triggers",
            "format_price,minimal fix",
            "--scope",
            "repo",
            "--cwd",
            str(tmp_path),
        ]
    )
    assert added.exit_code == 0, added.stdout

    listed = _run(["experience", "procedures", "list", "--scope", "repo", "--cwd", str(tmp_path)])
    assert listed.exit_code == 0, listed.stdout
    assert "general" in listed.stdout
    procedures_root = tmp_path / ".harness" / "procedures"
    assert procedures_root.is_dir()
    assert any((path / "procedure.json").is_file() for path in procedures_root.iterdir())


def test_experience_curate_dry_run_reports_actions(tmp_path: Path) -> None:
    _run(
        [
            "experience",
            "procedures",
            "add",
            "--name",
            "Scope discipline",
            "--body",
            "Keep the patch in the named formatter only.",
            "--triggers",
            "format_price",
            "--scope",
            "repo",
            "--cwd",
            str(tmp_path),
        ]
    )
    _run(
        [
            "experience",
            "procedures",
            "add",
            "--name",
            "Scope discipline duplicate",
            "--body",
            "Keep the patch in the named formatter only.",
            "--triggers",
            "format_price",
            "--scope",
            "repo",
            "--cwd",
            str(tmp_path),
        ]
    )

    curated = _run(
        [
            "experience",
            "curate",
            "--scope",
            "repo",
            "--cwd",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert curated.exit_code == 0, curated.stdout
    assert "duplicate" in curated.stdout
