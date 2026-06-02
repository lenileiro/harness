from harness.core.shell_feedback import shell_failure_hint


def test_shell_failure_hint_suggests_declared_runtime_after_editable_install_failure() -> None:
    hint = shell_failure_hint(
        "pip install -e .",
        exit_code=1,
        stdout="Defaulting to user installation because normal site-packages is not writeable",
        stderr=(
            'ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be '
            "installed in editable mode"
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


def test_shell_failure_hint_suggests_discovered_container_executable_path() -> None:
    hint = shell_failure_hint(
        "project-test-command",
        exit_code=127,
        stdout="",
        stderr="bash: line 1: project-test-command: command not found\n",
    )

    assert "containerized command" in hint
    assert "discover the tool inside a container" in hint
    assert "discovered executable path" in hint
    assert "instead of retrying the host command" in hint
