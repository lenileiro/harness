"""Standard-library worker sent to the selected host, container, or SSH host.

The process supervisor owns a POSIX process group. Its JSON control channel
keeps stdin available for interaction; EOF, deadline, and cancellation terminate
the group. This is process supervision, not an OS sandbox against hostile code
that deliberately detaches into a new session.
"""

from __future__ import annotations

import base64
import contextlib
import fnmatch
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def resolve(root: Path, raw: str) -> Path:
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        raise ValueError("path resolves outside the execution workspace")
    return target


def files(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["root"]).resolve(strict=True)
    operation = request["operation"]
    args = request["arguments"]
    cap = request["max_file_bytes"]
    path = resolve(root, args.get("path", "."))
    if operation in {"read_bytes", "write_bytes"}:
        relative = Path(args["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(part.casefold() == ".harness" for part in relative.parts)
        ):
            raise ValueError("transfer path must be a public relative workspace file")
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError("transfer path may not contain symlinks")
            if cursor.exists() and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("transfer path may not contain Windows reparse points")
        if operation == "read_bytes":
            if not path.is_file():
                raise ValueError("transfer path is not a regular file")
            with path.open("rb") as stream:
                data = stream.read(cap + 1)
            if len(data) > cap:
                raise ValueError("transfer file exceeds configured byte limit")
            return {
                "content": base64.b64encode(data).decode("ascii"),
                "metadata": {"bytes": len(data)},
            }
        data = base64.b64decode(args["data"], validate=True)
        if len(data) > cap:
            raise ValueError("transfer file exceeds configured byte limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"content": "transferred", "metadata": {"bytes": len(data)}}
    if operation == "probe":
        if not root.is_dir():
            raise ValueError("execution workspace is not a directory")
        return {"content": str(root), "metadata": {"cwd": str(root)}}
    if operation == "git_status":
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "-z"],
                cwd=root,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"content": "", "metadata": {"available": False}}
        return {
            "content": result.stdout.decode(errors="replace"),
            "metadata": {"available": result.returncode == 0},
        }
    if operation == "read_file":
        if not path.is_file():
            raise ValueError("path is not a regular file")
        with path.open("rb") as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("file exceeds configured byte limit")
        return {"content": raw.decode("utf-8"), "metadata": {"path": str(path), "bytes": len(raw)}}
    if operation in {"write_file", "edit_file"}:
        before = None
        if path.exists():
            if not path.is_file():
                raise ValueError("path is not a regular file")
            with path.open("rb") as stream:
                raw = stream.read(cap + 1)
            if len(raw) > cap:
                raise ValueError("existing file exceeds configured byte limit")
            before = raw.decode("utf-8")
        content = args.get("content")
        if operation == "edit_file":
            if before is None:
                raise ValueError("file does not exist")
            old = args["old"]
            if not old or before.count(old) != 1:
                raise ValueError("old text must occur exactly once")
            content = before.replace(old, args["new"], 1)
        if not isinstance(content, str) or len(content.encode("utf-8")) > cap:
            raise ValueError("content is invalid or exceeds configured byte limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "content": f"wrote {len(content.encode('utf-8'))} bytes to {path}",
            "metadata": {
                "path": str(path),
                "created": before is None,
                "content_before": before[:8192] if before is not None else None,
                "content_after": content[:8192],
            },
        }
    if operation == "list_dir":
        entries = []
        for entry in sorted(path.iterdir()):
            if not entry.resolve().is_relative_to(root):
                continue
            entries.append(entry.name + ("/" if entry.is_dir() else ""))
        return {
            "content": "\n".join(entries) or "(empty)",
            "metadata": {"path": str(path), "entries": len(entries)},
        }
    if operation == "glob":
        pattern = args["pattern"]
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("glob must be relative and may not contain '..'")
        result = []
        # Walk without following directory symlinks, then filter every canonical
        # target. No out-of-workspace traversal through symlink aliases.
        for directory, directories, names in os.walk(root, followlinks=False):
            for name in sorted(directories + names):
                entry = Path(directory) / name
                relative = entry.relative_to(root).as_posix()
                if not entry.resolve().is_relative_to(root):
                    continue
                if fnmatch.fnmatchcase(relative, pattern) or (
                    pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:])
                ):
                    result.append(relative)
                    if len(result) >= 500:
                        break
            if len(result) >= 500:
                break
        return {
            "content": "\n".join(sorted(result)) or "(no matches)",
            "metadata": {"matches": len(result), "capped": len(result) >= 500},
        }
    raise ValueError("unknown filesystem operation")


def kill_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0.25)
    # Also remove children after a shell exits successfully or on TERM.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def supervise_windows(request: dict[str, Any]) -> None:
    """Pipes use bounded daemon pumps because Windows select() accepts sockets only.

    The hosting transport assigns this worker's gated ancestor to a kill-on-close
    Job Object before supplying the request. All normal descendants inherit it.
    """
    import queue
    import threading

    if request.get("pty"):
        raise ValueError(
            "PTY is unavailable on native Windows; use interactive process input without PTY"
        )
    root = Path(request["root"]).resolve(strict=True)
    cwd = resolve(root, request.get("cwd", "."))
    process = subprocess.Popen(
        [request["shell"], "-c", request["command"]],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdin is not None and process.stdout is not None
    output: queue.Queue[bytes] = queue.Queue(maxsize=64)
    controls: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=16)
    inputs: queue.Queue[bytes] = queue.Queue(maxsize=16)
    reader_done = threading.Event()

    def read_output() -> None:
        try:
            assert process.stdout is not None
            while data := os.read(process.stdout.fileno(), 16384):
                output.put(data)
        except OSError:
            pass
        finally:
            reader_done.set()

    def read_controls() -> None:
        pending = bytearray()
        try:
            # Never hold Python's buffered stdin lock in a daemon thread: that
            # can stall interpreter shutdown while a descendant retains a pipe.
            while chunk := os.read(0, 65536):
                pending.extend(chunk)
                if len(pending) > 1048576:
                    raise ValueError("control input exceeds limit")
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    controls.put(json.loads(line))
        except (ValueError, OSError):
            pass
        finally:
            controls.put({"action": "disconnected"})

    def write_input() -> None:
        try:
            assert process.stdin is not None
            while True:
                data = inputs.get()
                process.stdin.write(data)
                process.stdin.flush()
        except (OSError, ValueError):
            pass

    for target in (read_output, read_controls, write_input):
        threading.Thread(target=target, daemon=True).start()
    reason = "exited"
    deadline = time.monotonic() + request["timeout"]
    emitted = 0
    truncated = False

    def drain_output() -> None:
        nonlocal emitted, truncated
        # Bounded batch: an output flood cannot starve deadline/control handling.
        for _ in range(64):
            try:
                data = output.get_nowait()
            except queue.Empty:
                break
            cap = request.get("max_output_bytes", 16777216)
            remaining = max(0, cap - emitted)
            if len(data) > remaining and not truncated:
                emit({"event": "truncated"})
                truncated = True
            data = data[:remaining]
            if data:
                emitted += len(data)
                emit({"event": "output", "data": base64.b64encode(data).decode("ascii")})

    try:
        emit({"event": "started", "pid": process.pid, "cwd": str(cwd)})
        while process.poll() is None:
            drain_output()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timed_out"
                break
            try:
                control = controls.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if control["action"] in {"terminate", "disconnected"}:
                reason = "terminated" if control["action"] == "terminate" else "disconnected"
                break
            if control["action"] == "input":
                data = base64.b64decode(control["data"], validate=True)
                if len(data) > 65536:
                    raise ValueError("process input exceeds 65536-byte limit")
                try:
                    inputs.put_nowait(data)
                except queue.Full:
                    raise ValueError("pending process input exceeds limit") from None
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        # A descendant can retain the pipe. The parent Job Object removes it when
        # this worker exits; never block here waiting for that descendant's EOF.
        drain_until = time.monotonic() + 0.2
        while not reader_done.is_set() and time.monotonic() < drain_until:
            drain_output()
            reader_done.wait(0.01)
        drain_output()
    emit({"event": "exit", "exit_code": process.returncode, "reason": reason})


def supervise(request: dict[str, Any]) -> None:
    if os.name == "nt":
        supervise_windows(request)
        return
    if os.name != "posix":
        raise ValueError("managed execution requires a POSIX or Windows backend")
    root = Path(request["root"]).resolve(strict=True)
    cwd = resolve(root, request.get("cwd", "."))
    master = None
    slave = None
    if request.get("pty"):
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()

        def acquire_terminal() -> None:
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        process = subprocess.Popen(
            [request["shell"], "-c", request["command"]],
            cwd=cwd,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            preexec_fn=acquire_terminal,
        )
        os.close(slave)
        read_fd = write_fd = master
    else:
        process = subprocess.Popen(
            [request["shell"], "-c", request["command"]],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None
        read_fd, write_fd = process.stdout.fileno(), process.stdin.fileno()
    deadline = time.monotonic() + request["timeout"]
    controls = bytearray()
    pending_input = bytearray()
    emitted_bytes = 0
    truncated = False
    reason = "exited"
    output_open = True
    os.set_blocking(read_fd, False)
    os.set_blocking(write_fd, False)

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)

    def emit_output(output: bytes) -> None:
        nonlocal emitted_bytes, truncated
        cap = request.get("max_output_bytes")
        if cap is not None:
            remaining = max(0, cap - emitted_bytes)
            if len(output) > remaining and not truncated:
                emit({"event": "truncated"})
                truncated = True
            output = output[:remaining]
        if output:
            emitted_bytes += len(output)
            emit({"event": "output", "data": base64.b64encode(output).decode("ascii")})

    try:
        emit({"event": "started", "pid": process.pid, "cwd": str(cwd)})
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timed_out"
                break
            ready, writable, _ = select.select(
                [0, read_fd] if output_open else [0],
                [write_fd] if pending_input else [],
                [],
                min(remaining, 0.1),
            )
            if writable:
                try:
                    written = os.write(write_fd, pending_input[:16384])
                    del pending_input[:written]
                except BlockingIOError:
                    pass
                except BrokenPipeError:
                    pending_input.clear()
            if read_fd in ready:
                try:
                    output = os.read(read_fd, 16384)
                except OSError:
                    output = b""
                if output:
                    emit_output(output)
                else:
                    output_open = False
            if 0 in ready:
                incoming = os.read(0, 65536)
                if not incoming:
                    reason = "disconnected"
                    break
                controls.extend(incoming)
                while b"\n" in controls:
                    line, _, rest = controls.partition(b"\n")
                    controls = bytearray(rest)
                    control = json.loads(line)
                    if control["action"] == "terminate":
                        reason = "terminated"
                        break
                    if control["action"] == "input":
                        pending_input.extend(base64.b64decode(control["data"]))
                        if len(pending_input) > 1048576:
                            raise ValueError("pending process input exceeds limit")
                if reason == "terminated":
                    break
    finally:
        kill_group(process)
        # Drain already-buffered tail after group cleanup; never wait on a
        # detached descendant retaining stdout.
        while True:
            try:
                output = os.read(read_fd, 16384)
            except (BlockingIOError, OSError):
                break
            if not output:
                break
            emit_output(output)
        if master is not None:
            os.close(master)
        elif process.stdout is not None and process.stdin is not None:
            process.stdout.close()
            process.stdin.close()
    emit({"event": "exit", "exit_code": process.returncode, "reason": reason})


def main() -> None:
    request = json.loads(base64.b64decode(sys.stdin.buffer.readline()))
    try:
        if request["operation"] == "shell":
            supervise(request)
        else:
            emit({"event": "result", **files(request)})
    except Exception as exc:
        emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
