import json
import plistlib
import subprocess
import sys
import time

from typer.testing import CliRunner

from harness.cli.__main__ import app
from harness.cli.service_commands import ServiceSpec, _argv, boot_manifest


def test_owned_service_start_dedup_stop_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    runner = CliRunner()
    script = tmp_path / "worker.py"
    script.write_text("import time\nprint('service ready', flush=True)\ntime.sleep(60)\n")
    result = runner.invoke(
        app,
        ["service", "install", "worker", "--cwd", str(tmp_path), "--", sys.executable, str(script)],
    )
    assert result.exit_code == 0, result.output
    try:
        result = runner.invoke(app, ["service", "start", "worker"])
        assert result.exit_code == 0, result.output
        first = runner.invoke(app, ["service", "status", "worker"])
        state = json.loads(first.output)
        assert state["alive"]
        result = runner.invoke(app, ["service", "start", "worker"])
        assert "already running" in result.output
        assert (
            json.loads(runner.invoke(app, ["service", "status", "worker"]).output)["pid"]
            == state["pid"]
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = runner.invoke(app, ["service", "logs", "worker"])
            if "service ready" in result.output:
                break
            time.sleep(0.05)
        assert "service ready" in result.output
    finally:
        result = runner.invoke(app, ["service", "stop", "worker"])
        assert result.exit_code == 0, result.output
    state = json.loads(runner.invoke(app, ["service", "status", "worker"]).output)
    assert not state["alive"] and state["status"] == "stopped"


def test_crash_is_bounded_and_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "service",
            "install",
            "crash",
            "--max-restarts",
            "0",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
    )
    assert result.exit_code == 0, result.output
    completed = subprocess.run(_argv("crash"), capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr.decode()
    state = json.loads(runner.invoke(app, ["service", "status", "crash"]).output)
    assert state["status"] == "failed" and state["exit_code"] == 7 and state["restarts"] == 0


def test_boot_manifests_preserve_arguments_and_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "identity"))
    spec = ServiceSpec(name="test", command=["harness", "scheduler", "start"], cwd=tmp_path)
    argv = ["/path with space/python", "-m", "harness.cli", "service", "supervise", "test"]
    _, content = boot_manifest(spec, "darwin", argv=argv, log=tmp_path / "log")
    assert plistlib.loads(content)["ProgramArguments"] == argv
    _, content = boot_manifest(spec, "linux", argv=argv, log=tmp_path / "log")
    assert 'ExecStart="/path with space/python" "-m"' in content.decode()
    _, content = boot_manifest(spec, "win32", argv=argv, log=tmp_path / "log")
    assert "<RunLevel>LeastPrivilege</RunLevel>" in content.decode("utf-16")


def test_boot_argv_restores_custom_profile_without_inherited_environment(tmp_path, monkeypatch):
    import os

    from harness.cli.profiles import activate_profile, create_profile, profile_root

    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "custom-catalog"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    profile = create_profile("boot")
    (profile_root("boot") / "credentials.env").write_text("SERVICE_TEST_TOKEN='configured'\n")
    script = tmp_path / "worker.py"
    marker = profile.workspace / "proof.json"
    script.write_text(
        "import json,os,pathlib\npathlib.Path('proof.json').write_text(json.dumps({'cwd':os.getcwd(),'credential_present':bool(os.environ.get('SERVICE_TEST_TOKEN'))}))\n"
    )
    result = CliRunner().invoke(
        app,
        ["--profile", "boot", "service", "install", "boot-test", "--", sys.executable, str(script)],
    )
    assert result.exit_code == 0, result.output
    with activate_profile("boot"):
        argv = _argv("boot-test")
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("HARNESS_")
    }
    completed = subprocess.run(argv, env=environment, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr.decode()
    assert json.loads(marker.read_text()) == {
        "cwd": str(profile.workspace),
        "credential_present": True,
    }


def test_normal_exit_cleans_up_background_children(tmp_path, monkeypatch):
    import os

    import psutil
    import pytest

    if os.name == "nt":
        pytest.skip(
            "POSIX process group contract; native Windows job ownership needs platform validation"
        )
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import subprocess,sys,pathlib,time\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\npathlib.Path('child.pid').write_text(str(p.pid))\ntime.sleep(.3)\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["service", "install", "orphan", "--cwd", str(tmp_path), "--", sys.executable, str(worker)],
    )
    assert result.exit_code == 0, result.output
    child = None
    try:
        result = subprocess.run(_argv("orphan"), capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr.decode()
        child_pid = int((tmp_path / "child.pid").read_text())
        try:
            child = psutil.Process(child_pid)
        except psutil.NoSuchProcess:
            return
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if child is not None and child.is_running():
            child.kill()


def test_stop_succeeds_when_the_group_holds_no_live_members(tmp_path, monkeypatch):
    """Teardown signals the group unconditionally; an empty one must not fail.

    Darwin reports EPERM (not ESRCH) for a group left holding only zombies, which
    is why teardown suppresses both. Here the worker's background child exits
    immediately, so nothing in the group is alive by the time the supervisor
    cleans up.
    """

    import os

    import pytest

    if os.name == "nt":
        pytest.skip("POSIX process group contract")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    worker = tmp_path / "short_worker.py"
    worker.write_text(
        "import subprocess,sys,pathlib,time\n"
        "p=subprocess.Popen([sys.executable,'-c','pass'])\n"
        "p.wait()\n"
        "pathlib.Path('child.pid').write_text(str(p.pid))\n"
        "time.sleep(.2)\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["service", "install", "brief", "--cwd", str(tmp_path), "--", sys.executable, str(worker)],
    )
    assert result.exit_code == 0, result.output

    completed = subprocess.run(_argv("brief"), capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr.decode()
    assert (tmp_path / "child.pid").exists()
