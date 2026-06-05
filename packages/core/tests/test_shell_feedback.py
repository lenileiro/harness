from pathlib import Path

from harness.core.shell_feedback import shell_failure_hint


def test_shell_failure_hint_suggests_declared_runtime_after_package_install_failure() -> None:
    hint = shell_failure_hint(
        "project install --editable .",
        exit_code=1,
        stdout="Defaulting to user installation because normal site-packages is not writeable",
        stderr=(
            "ERROR: Package manifests not found. Directory cannot be installed in editable mode"
        ),
    )

    assert "declared runtime files" in hint
    assert "environment/Dockerfile" in hint
    assert "Confirm the setup" in hint


def test_shell_failure_hint_suggests_import_check_after_unknown_package_install() -> None:
    hint = shell_failure_hint(
        "pip install .",
        exit_code=0,
        stdout="Successfully installed UNKNOWN-0.0.0",
        stderr="",
    )

    assert "did not produce a usable project install" in hint
    assert "importing the project package" in hint


def test_shell_feedback_has_no_hard_coded_python_packaging_files() -> None:
    source = Path(shell_failure_hint.__code__.co_filename).read_text(encoding="utf-8").lower()

    assert "setup.py" not in source
    assert "setup.cfg" not in source
    assert "no module named build" not in source


def test_shell_failure_hint_suggests_diagnostics_for_terse_test_failure() -> None:
    hint = shell_failure_hint(
        "./tests/run.sh",
        exit_code=1,
        stdout="",
        stderr="not ok: basic behavior\n",
    )

    assert "diagnostic hint" in hint
    assert "execution tracing" in hint
    assert "variable values" in hint


def test_shell_failure_hint_suggests_tracing_empty_shell_script_failure() -> None:
    hint = shell_failure_hint(
        "./tests/run.sh",
        exit_code=1,
        stdout="",
        stderr="",
    )

    assert "empty failure" in hint
    assert "bash -x ./tests/run.sh" in hint
    assert "Do not append `; echo $?`" in hint


def test_shell_failure_hint_suggests_tracing_make_invoked_script() -> None:
    hint = shell_failure_hint(
        "make test",
        exit_code=2,
        stdout="./tests/run.sh\n",
        stderr="make: *** [test] Error 1\n",
    )

    assert "diagnostic hint" in hint
    assert "Makefile target failed" in hint
    assert "bash -x ./tests/run.sh" in hint


def test_shell_failure_hint_explains_errexit_expected_failure_trace() -> None:
    hint = shell_failure_hint(
        "bash -x tests/run.sh",
        exit_code=1,
        stdout=(
            "+ set -euo pipefail\n"
            "+ '[' second = second ']'\n"
            "++ ./bin/ini_get server does_not_exist /tmp/tmp.ini\n"
            "++ true\n"
            "+ out=\n"
            "+ ./bin/ini_get server does_not_exist /tmp/tmp.ini\n"
            "+ rm -f /tmp/tmp.ini\n"
        ),
        stderr="",
    )

    assert "expected-failing command ran bare" in hint
    assert "set -e" in hint
    assert "command || rc=$?" in hint


def test_shell_failure_hint_does_not_flag_captured_errexit_expected_failure() -> None:
    hint = shell_failure_hint(
        "bash -x tests/run.sh",
        exit_code=1,
        stdout=(
            "+ set -euo pipefail\n"
            "+ ./bin/ini_get server does_not_exist fixture.ini\n"
            "+ rc=1\n"
            "+ '[' 1 -ne 0 ']'\n"
            "+ '[' -z '' ']'\n"
            "+ false\n"
        ),
        stderr="",
    )

    assert "expected-failing command ran bare" not in hint


def test_shell_failure_hint_suggests_discovered_container_executable_path() -> None:
    hint = shell_failure_hint(
        "project-test-command",
        exit_code=127,
        stdout="",
        stderr="bash: line 1: project-test-command: command not found\n",
    )

    assert "containerized command" in hint
    assert "discover the tool inside a container" in hint
    assert "same shell form" in hint
    assert "reset PATH" in hint
    assert "absolute executable path" in hint
    assert "instead of retrying the host command" in hint
