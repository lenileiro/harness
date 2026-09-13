"""Owned service supervision with durable status, crash limits, and boot manifests."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Annotated
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape

import psutil
import typer
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

from harness.core.paths import user_home

service_app = typer.Typer(
    help="Install and supervise owned background services.", no_args_is_help=True
)


class ServiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    command: list[str] = Field(min_length=1)
    cwd: Path
    max_restarts: int = Field(default=5, ge=0, le=100)
    restart_delay: float = Field(default=2, ge=0.1, le=300)


def _root(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", name):
        raise typer.BadParameter("Use a short lowercase service name")
    return user_home() / "services" / name


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _spec(name: str) -> ServiceSpec:
    try:
        spec = ServiceSpec.model_validate_json((_root(name) / "service.json").read_text())
        if spec.name != name:
            raise ValueError("name mismatch")
        return spec
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"Service {name} is not installed or its configuration is invalid"
        ) from exc


def _state(name: str) -> dict:
    try:
        return json.loads((_root(name) / "status.json").read_text())
    except (OSError, ValueError):
        return {"status": "stopped"}


def _process(state: dict) -> psutil.Process | None:
    try:
        process = psutil.Process(state["pid"])
        if (
            abs(process.create_time() - state["created_at"]) < 0.001
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return process
    except (psutil.Error, KeyError, TypeError):
        pass
    return None


def _argv(name: str) -> list[str]:
    from harness.cli.profiles import profiles_root

    # Boot environments do not inherit terminal variables. Bind catalog and
    # state paths in argv on every OS, including Task Scheduler on Windows.
    argv = [
        sys.executable,
        "-m",
        "harness.cli",
        "--home",
        str(user_home()),
        "--profiles-root",
        str(profiles_root()),
    ]
    if profile := os.environ.get("HARNESS_PROFILE"):
        argv.extend(["--profile", profile])
    return [*argv, "service", "supervise", name]


def _label(name: str) -> str:
    identity = hashlib.sha256(str(user_home()).encode()).hexdigest()[:10]
    return f"dev.harness.{identity}.{name}"


def boot_manifest(
    spec: ServiceSpec, platform: str, *, argv: list[str], log: Path
) -> tuple[str, bytes]:
    """Generate inspectable user-scoped boot integration without enabling it."""
    label = _label(spec.name)
    if platform == "darwin":
        return "launchd.plist", plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": argv,
                "WorkingDirectory": str(spec.cwd),
                "RunAtLoad": True,
                "StandardOutPath": str(log),
                "StandardErrorPath": str(log),
                "EnvironmentVariables": {"HARNESS_HOME": str(user_home())},
            }
        )
    if platform == "win32":
        import subprocess as sp

        command, arguments = xml_escape(argv[0]), xml_escape(sp.list2cmdline(argv[1:]))
        xml = f'<?xml version="1.0" encoding="UTF-16"?><Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers><Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings><Actions Context="Author"><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments><WorkingDirectory>{xml_escape(str(spec.cwd))}</WorkingDirectory></Exec></Actions></Task>'
        return "task.xml", xml.encode("utf-16")

    def quote(value: str) -> str:
        return (
            '"'
            + value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("%", "%%")
            .replace("\n", "\\n")
            + '"'
        )

    unit = f"[Unit]\nDescription=Harness {spec.name}\nAfter=network-online.target\n[Service]\nType=simple\nWorkingDirectory={quote(str(spec.cwd))}\nEnvironment={quote('HARNESS_HOME=' + str(user_home()))}\nExecStart={' '.join(quote(arg) for arg in argv)}\nRestart=no\n[Install]\nWantedBy=default.target\n"
    return "systemd.service", unit.encode()


@service_app.command(
    "install", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def install(
    ctx: typer.Context,
    name: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_restarts: int = typer.Option(5, "--max-restarts"),
) -> None:
    """Install a command after --, e.g. service install routines -- harness scheduler start."""
    command = list(ctx.args)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise typer.BadParameter("Provide the service command after --")
    if any("\x00" in value for value in command):
        raise typer.BadParameter("Command arguments must not contain NUL")
    spec = ServiceSpec(
        name=name, command=command, cwd=(cwd or Path.cwd()).resolve(), max_restarts=max_restarts
    )
    root = _root(name)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with FileLock(root / "control.lock"):
        if (root / "service.json").exists():
            raise typer.BadParameter("Service already installed; use a different name")
        _write(root / "service.json", spec.model_dump(mode="json"))
        manifest_name, content = boot_manifest(
            spec, sys.platform, argv=_argv(name), log=root / "output.log"
        )
        (root / manifest_name).write_bytes(content)
    typer.echo(f"Installed {name}. Boot manifest: {root / manifest_name}")


@service_app.command("start")
def start(name: str) -> None:
    spec, root = _spec(name), _root(name)
    with FileLock(root / "control.lock"):
        if _process(_state(name)):
            typer.echo(f"{name} is already running")
            return
        (root / "stop").unlink(missing_ok=True)
        with (root / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                _argv(name),
                cwd=spec.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
                if os.name == "nt"
                else 0,
            )
        # Long-lived callers (tests/embedded CLI) must reap their detached child.
        threading.Thread(target=process.wait, daemon=True, name=f"harness-{name}-reaper").start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if _process(_state(name)):
                typer.echo(f"Started {name}")
                return
            if process.poll() is not None:
                state = _state(name)
                if state.get("pid") == process.pid and state.get("status") == "exited":
                    typer.echo(f"{name} completed")
                    return
                raise typer.BadParameter(f"Supervisor failed; see {root / 'supervisor.log'}")
            time.sleep(0.05)
        raise typer.BadParameter(f"Supervisor did not become ready; inspect service logs {name}")


@service_app.command("supervise", hidden=True)
def supervise(name: str) -> None:
    import signal

    spec, root = _spec(name), _root(name)
    lock = FileLock(root / "running.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return
    child: subprocess.Popen | None = None
    descendants: dict[int, psutil.Process] = {}

    def stop_child() -> None:
        if child is None:
            return
        # A service command may exit while its subprocesses still run. Give
        # each attempt an owned process group and clean it before any restart.
        if os.name != "nt":
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGTERM)
        for process in descendants.values():
            with suppress(psutil.NoSuchProcess):
                process.terminate()
        if child.poll() is None:
            child.terminate()
        _, alive = psutil.wait_procs(list(descendants.values()), timeout=3)
        for process in alive:
            with suppress(psutil.NoSuchProcess):
                process.kill()
        if os.name != "nt":
            # By now the group may be empty or hold only zombies: that is ESRCH
            # on Linux and EPERM on Darwin, and neither is actionable.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGKILL)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        descendants.clear()

    def request_stop(*_):
        (root / "stop").touch(mode=0o600)

    previous = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    identity = {"pid": os.getpid(), "created_at": psutil.Process().create_time()}
    try:
        for attempt in range(spec.max_restarts + 1):
            if (root / "stop").exists():
                break
            _write(root / "status.json", {**identity, "status": "starting", "restarts": attempt})
            log_path = root / "output.log"
            if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
                log_path.replace(root / "output.previous.log")
            try:
                with log_path.open("ab") as output:
                    child = subprocess.Popen(
                        spec.command,
                        cwd=spec.cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=os.name != "nt",
                    )
            except OSError as exc:
                _write(
                    root / "status.json",
                    {
                        **identity,
                        "status": "failed",
                        "error": type(exc).__name__,
                        "restarts": attempt,
                    },
                )
                return
            _write(
                root / "status.json",
                {**identity, "status": "running", "child_pid": child.pid, "restarts": attempt},
            )
            while child.poll() is None and not (root / "stop").exists():
                with suppress(psutil.NoSuchProcess):
                    descendants.update(
                        (process.pid, process)
                        for process in psutil.Process(child.pid).children(recursive=True)
                    )
                time.sleep(0.1)
            if (root / "stop").exists():
                break
            code = child.wait()
            stop_child()
            child = None
            _write(
                root / "status.json",
                {
                    **identity,
                    "status": "exited" if code == 0 else "crashed",
                    "exit_code": code,
                    "restarts": attempt,
                },
            )
            if code == 0:
                return
            if attempt == spec.max_restarts:
                _write(
                    root / "status.json",
                    {**identity, "status": "failed", "exit_code": code, "restarts": attempt},
                )
                return
            deadline = time.monotonic() + min(60, spec.restart_delay * 2**attempt)
            while time.monotonic() < deadline and not (root / "stop").exists():
                time.sleep(0.1)
    finally:
        stop_child()
        if (root / "stop").exists():
            _write(root / "status.json", {**identity, "status": "stopped"})
        lock.release()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@service_app.command("status")
def status(name: str) -> None:
    _spec(name)
    state = _state(name)
    state["alive"] = _process(state) is not None
    if not state["alive"] and state.get("status") in {"starting", "running", "crashed"}:
        state["status"] = "interrupted"
    typer.echo(json.dumps(state, indent=2))


@service_app.command("stop")
def stop(name: str) -> None:
    _spec(name)
    root = _root(name)
    with FileLock(root / "control.lock"):
        (root / "stop").touch(mode=0o600)
        process = _process(_state(name))
        if process:
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                raise typer.BadParameter(
                    "Supervisor did not stop; inspect status and logs"
                ) from None
    typer.echo(f"Stopped {name}")


@service_app.command("restart")
def restart(name: str) -> None:
    stop(name)
    start(name)


@service_app.command("logs")
def logs(name: str, lines: Annotated[int, typer.Option("--lines", min=1, max=10000)] = 100) -> None:
    _spec(name)
    path = _root(name) / "output.log"
    if path.exists():
        from collections import deque

        with path.open(errors="replace") as source:
            typer.echo("".join(deque(source, maxlen=lines)), nl=False)


@service_app.command("enable-boot")
def enable_boot(name: str) -> None:
    """Register the generated manifest with the current user's OS service manager."""
    import shutil

    _spec(name)
    root = _root(name)
    label = _label(name)
    if sys.platform == "darwin":
        destination = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "launchd.plist", destination)
        command = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)]
    elif os.name == "nt":
        command = ["schtasks", "/Create", "/TN", label, "/XML", str(root / "task.xml")]
    else:
        destination = Path.home() / ".config/systemd/user" / f"{label}.service"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "systemd.service", destination)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        command = ["systemctl", "--user", "enable", destination.name]
    subprocess.run(command, check=True)
    typer.echo(f"Enabled boot integration for {name}")


@service_app.command("disable-boot")
def disable_boot(name: str) -> None:
    """Remove only this profile's service from the current user's boot manager."""
    _spec(name)
    label = _label(name)
    if sys.platform == "darwin":
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], check=True)
        (Path.home() / "Library/LaunchAgents" / f"{label}.plist").unlink(missing_ok=True)
    elif os.name == "nt":
        subprocess.run(["schtasks", "/Delete", "/TN", label, "/F"], check=True)
    else:
        subprocess.run(["systemctl", "--user", "disable", f"{label}.service"], check=True)
        (Path.home() / ".config/systemd/user" / f"{label}.service").unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    typer.echo(f"Disabled boot integration for {name}")
