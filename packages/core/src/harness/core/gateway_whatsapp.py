from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from harness.core.gateway_whatsapp_assets import (
    WHATSAPP_BRIDGE_JS,
    WHATSAPP_BRIDGE_PACKAGE_JSON,
)

DEFAULT_GATEWAY_CHILD_TIMEOUT_SECONDS = 600


@dataclass(slots=True)
class WhatsAppBridgeConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "gemma4:latest"
    mode: str = "self-chat"
    allowed_users: list[str] = field(default_factory=list)
    bridge_port: int = 8741
    reply_prefix: str = "Harness Agent\n────────────\n"
    max_gateway_concurrency: int = 1
    max_gateway_queue: int = 3
    gateway_child_timeout_seconds: int = DEFAULT_GATEWAY_CHILD_TIMEOUT_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WhatsAppBridgeStatus:
    config: WhatsAppBridgeConfig
    root: Path
    project_dir: Path
    session_dir: Path
    log_path: Path
    paired: bool
    dependencies_installed: bool
    bridge_running: bool
    bridge_connected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "model": self.config.model,
            "mode": self.config.mode,
            "allowed_users": list(self.config.allowed_users),
            "bridge_port": self.config.bridge_port,
            "reply_prefix": self.config.reply_prefix,
            "max_gateway_concurrency": self.config.max_gateway_concurrency,
            "max_gateway_queue": self.config.max_gateway_queue,
            "gateway_child_timeout_seconds": self.config.gateway_child_timeout_seconds,
            "root": str(self.root),
            "project_dir": str(self.project_dir),
            "session_dir": str(self.session_dir),
            "log_path": str(self.log_path),
            "paired": self.paired,
            "dependencies_installed": self.dependencies_installed,
            "bridge_running": self.bridge_running,
            "bridge_connected": self.bridge_connected,
        }


def default_whatsapp_root(cwd: Path) -> Path:
    return cwd / ".harness" / "gateway" / "whatsapp"


def default_whatsapp_project_dir(cwd: Path) -> Path:
    return default_whatsapp_root(cwd) / "bridge"


def default_whatsapp_session_dir(cwd: Path) -> Path:
    return default_whatsapp_root(cwd) / "session"


def default_whatsapp_log_path(cwd: Path) -> Path:
    return default_whatsapp_root(cwd) / "bridge.log"


def default_whatsapp_config_path(cwd: Path) -> Path:
    return default_whatsapp_root(cwd) / "config.json"


def load_whatsapp_bridge_config(cwd: Path) -> WhatsAppBridgeConfig:
    path = default_whatsapp_config_path(cwd)
    if not path.exists():
        return WhatsAppBridgeConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return WhatsAppBridgeConfig(
        enabled=bool(payload.get("enabled", False)),
        provider=str(payload.get("provider", "ollama")).strip() or "ollama",
        model=str(payload.get("model", "gemma4:latest")).strip() or "gemma4:latest",
        mode=str(payload.get("mode", "self-chat")).strip() or "self-chat",
        allowed_users=[
            str(item).strip() for item in payload.get("allowed_users", []) if str(item).strip()
        ],
        bridge_port=int(payload.get("bridge_port", 8741)),
        reply_prefix=str(payload.get("reply_prefix", "Harness Agent\n────────────\n")),
        max_gateway_concurrency=max(1, int(payload.get("max_gateway_concurrency", 1))),
        max_gateway_queue=max(0, int(payload.get("max_gateway_queue", 3))),
        gateway_child_timeout_seconds=max(
            5,
            int(
                payload.get(
                    "gateway_child_timeout_seconds",
                    DEFAULT_GATEWAY_CHILD_TIMEOUT_SECONDS,
                )
            ),
        ),
    )


def save_whatsapp_bridge_config(cwd: Path, config: WhatsAppBridgeConfig) -> Path:
    path = default_whatsapp_config_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def ensure_whatsapp_bridge_project(cwd: Path) -> Path:
    project_dir = default_whatsapp_project_dir(cwd)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "package.json").write_text(WHATSAPP_BRIDGE_PACKAGE_JSON, encoding="utf-8")
    bridge_path = project_dir / "bridge.js"
    bridge_path.write_text(WHATSAPP_BRIDGE_JS, encoding="utf-8")
    bridge_path.chmod(0o755)
    return project_dir


def _active_whatsapp_bridge_path(cwd: Path) -> Path:
    return default_whatsapp_session_dir(cwd) / "active-bridge.json"


def _read_active_whatsapp_bridge_pid(cwd: Path) -> int | None:
    path = _active_whatsapp_bridge_path(cwd)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        pid = int(payload.get("pid", 0))
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_whatsapp_bridge_command(cwd: Path, command: str) -> bool:
    bridge_script = default_whatsapp_project_dir(cwd) / "bridge.js"
    bridge_paths = {str(bridge_script), str(bridge_script.resolve())}
    return any(path in command for path in bridge_paths)


def _discover_whatsapp_bridge_pids(cwd: Path) -> list[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    current_pid = os.getpid()
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid <= 0 or pid == current_pid:
            continue
        if _is_whatsapp_bridge_command(cwd, parts[1]):
            pids.append(pid)
    return pids


def _stop_whatsapp_bridge_pid(pid: int, *, grace_seconds: float) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, PermissionError):
        return False

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.05)

    with suppress(OSError, PermissionError):
        os.kill(pid, signal.SIGKILL)
    return True


def stop_stale_whatsapp_bridge(cwd: Path, *, grace_seconds: float = 1.0) -> bool:
    pid = _read_active_whatsapp_bridge_pid(cwd)
    stale_pids: list[int] = []
    if pid and pid != os.getpid() and _is_whatsapp_bridge_command(cwd, _process_command(pid)):
        stale_pids.append(pid)
    for discovered_pid in _discover_whatsapp_bridge_pids(cwd):
        if discovered_pid not in stale_pids:
            stale_pids.append(discovered_pid)

    stopped = False
    for stale_pid in stale_pids:
        stopped = _stop_whatsapp_bridge_pid(stale_pid, grace_seconds=grace_seconds) or stopped
    return stopped


def install_whatsapp_bridge_dependencies(
    cwd: Path,
    *,
    npm_bin: str = "npm",
) -> Path:
    project_dir = ensure_whatsapp_bridge_project(cwd)
    subprocess.run(
        [npm_bin, "install", "--no-fund", "--no-audit", "--progress=false"],
        cwd=str(project_dir),
        check=True,
    )
    return project_dir


def clear_whatsapp_session(cwd: Path) -> Path:
    session_dir = default_whatsapp_session_dir(cwd)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def is_whatsapp_paired(cwd: Path) -> bool:
    return (default_whatsapp_session_dir(cwd) / "creds.json").exists()


def build_whatsapp_bridge_env(cwd: Path) -> dict[str, str]:
    config = load_whatsapp_bridge_config(cwd)
    dotenv_path = cwd / ".env"
    shell_timeout_seconds = max(1, min(30, config.gateway_child_timeout_seconds - 5))
    return {
        "HARNESS_WHATSAPP_MODE": config.mode,
        "HARNESS_WHATSAPP_ALLOWED_USERS": ",".join(config.allowed_users),
        "HARNESS_WHATSAPP_REPLY_PREFIX": config.reply_prefix,
        "HARNESS_WHATSAPP_WORKSPACE_CWD": str(cwd.resolve()),
        "HARNESS_WHATSAPP_UV_BIN": shutil.which("uv") or "uv",
        "HARNESS_WHATSAPP_ENV_FILE": str(dotenv_path.resolve()) if dotenv_path.is_file() else "",
        "HARNESS_WHATSAPP_MAX_CONCURRENCY": str(config.max_gateway_concurrency),
        "HARNESS_WHATSAPP_MAX_QUEUE": str(config.max_gateway_queue),
        "HARNESS_WHATSAPP_CHILD_TIMEOUT_MS": str(config.gateway_child_timeout_seconds * 1000),
        "HARNESS_SHELL_DEFAULT_TIMEOUT": str(shell_timeout_seconds),
    }


def run_whatsapp_pairing(
    cwd: Path,
    *,
    node_bin: str = "node",
) -> subprocess.CompletedProcess[str]:
    project_dir = ensure_whatsapp_bridge_project(cwd)
    session_dir = default_whatsapp_session_dir(cwd)
    session_dir.mkdir(parents=True, exist_ok=True)
    config = load_whatsapp_bridge_config(cwd)
    env = build_whatsapp_bridge_env(cwd)
    runtime_env = {**os.environ, **env}
    return subprocess.run(
        [
            node_bin,
            str(project_dir / "bridge.js"),
            "--pair-only",
            "--session",
            str(session_dir),
            "--mode",
            config.mode,
        ],
        cwd=str(project_dir),
        env=runtime_env,
        check=True,
        text=True,
    )


def start_whatsapp_bridge(
    cwd: Path,
    *,
    node_bin: str = "node",
) -> subprocess.CompletedProcess[str]:
    project_dir = ensure_whatsapp_bridge_project(cwd)
    session_dir = default_whatsapp_session_dir(cwd)
    session_dir.mkdir(parents=True, exist_ok=True)
    stop_stale_whatsapp_bridge(cwd)
    config = load_whatsapp_bridge_config(cwd)
    env = build_whatsapp_bridge_env(cwd)
    runtime_env = {**os.environ, **env}
    return subprocess.run(
        [
            node_bin,
            str(project_dir / "bridge.js"),
            "--port",
            str(config.bridge_port),
            "--session",
            str(session_dir),
            "--mode",
            config.mode,
        ],
        cwd=str(project_dir),
        env=runtime_env,
        check=True,
        text=True,
    )


def probe_whatsapp_bridge(
    cwd: Path,
    *,
    timeout: float = 2.0,
) -> dict[str, Any] | None:
    config = load_whatsapp_bridge_config(cwd)
    url = f"http://127.0.0.1:{config.bridge_port}/health"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected_cwd = str(cwd.resolve())
    reported_cwd = str(payload.get("workspace_cwd", "")).strip()
    if reported_cwd and reported_cwd != expected_cwd:
        return None
    if not reported_cwd:
        return None
    return payload


def read_whatsapp_bridge_status(cwd: Path) -> WhatsAppBridgeStatus:
    config = load_whatsapp_bridge_config(cwd)
    project_dir = default_whatsapp_project_dir(cwd)
    session_dir = default_whatsapp_session_dir(cwd)
    log_path = default_whatsapp_log_path(cwd)
    health = probe_whatsapp_bridge(cwd)
    return WhatsAppBridgeStatus(
        config=config,
        root=default_whatsapp_root(cwd),
        project_dir=project_dir,
        session_dir=session_dir,
        log_path=log_path,
        paired=is_whatsapp_paired(cwd),
        dependencies_installed=(project_dir / "node_modules").exists(),
        bridge_running=health is not None,
        bridge_connected=bool(health and health.get("status") == "connected"),
    )


def send_whatsapp_text_message(
    *,
    cwd: Path | None = None,
    to: str,
    text: str,
    reply_to: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    working_dir = (cwd or Path.cwd()).resolve()
    config = load_whatsapp_bridge_config(working_dir)
    if probe_whatsapp_bridge(working_dir, timeout=min(2.0, timeout)) is None:
        raise ConnectionError("whatsapp bridge is not running for this workspace")
    url = f"http://127.0.0.1:{config.bridge_port}/send"
    body: dict[str, Any] = {
        "chatId": to,
        "message": text,
    }
    if reply_to:
        body["replyTo"] = reply_to
    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


__all__ = [
    "WhatsAppBridgeConfig",
    "WhatsAppBridgeStatus",
    "build_whatsapp_bridge_env",
    "clear_whatsapp_session",
    "default_whatsapp_config_path",
    "default_whatsapp_log_path",
    "default_whatsapp_project_dir",
    "default_whatsapp_root",
    "default_whatsapp_session_dir",
    "ensure_whatsapp_bridge_project",
    "install_whatsapp_bridge_dependencies",
    "is_whatsapp_paired",
    "load_whatsapp_bridge_config",
    "probe_whatsapp_bridge",
    "read_whatsapp_bridge_status",
    "run_whatsapp_pairing",
    "save_whatsapp_bridge_config",
    "send_whatsapp_text_message",
    "start_whatsapp_bridge",
    "stop_stale_whatsapp_bridge",
]
