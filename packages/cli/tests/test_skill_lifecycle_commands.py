import json

from typer.testing import CliRunner

from harness.cli.__main__ import app


def test_operator_can_inspect_update_remove_and_rollback(tmp_path):
    runner = CliRunner()
    source = tmp_path / "source/check-output"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: check-output\ndescription: Check output\n---\nOld instructions\n"
    )

    def run(*args):
        return runner.invoke(app, ["skills", *args, "--cwd", str(tmp_path)])

    assert run("install", str(source)).exit_code == 0
    inspected = run("inspect", "check-output")
    assert inspected.exit_code == 0, inspected.output
    first = json.loads(inspected.output)["revision"]
    (source / "SKILL.md").write_text(
        "---\nname: check-output\ndescription: Check output\n---\nNew instructions\n"
    )
    updated = run("update", "check-output", "--source", str(source), "--expected", first)
    assert updated.exit_code == 0, updated.output
    second = json.loads(updated.output)["revision"]
    stale = run("remove", "check-output", "--expected", first)
    assert stale.exit_code == 2 and "changed" in stale.output
    assert run("remove", "check-output", "--expected", second).exit_code == 0
    recovered = run("rollback", "check-output", first, "--expected", "absent")
    assert recovered.exit_code == 0, recovered.output
    assert "Old instructions" in run("show", "check-output").output


def test_install_git_rejects_mutable_revision_before_network(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "skills",
            "install-git",
            "check-output",
            "https://github.com/o/r",
            "--commit",
            "main",
            "--path",
            ".",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2 and "40-character" in result.output
