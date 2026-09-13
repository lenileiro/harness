# Managed execution tools

```python
from pathlib import Path
from harness.tools.execution import ExecutionConfig, ExecutionToolset

config = ExecutionConfig(backend="docker", docker_image="python:3.12-slim")
async with ExecutionToolset(config, cwd=Path.cwd()) as execution:
    tools = execution.tools
    # Register these instead of the built-in filesystem and shell tools.
    # Keep this context open across model calls that share process handles.
```

The toolset provides `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`,
`shell`, `verify_work`, and `process`. File operations execute on the configured backend.
`verify_work` requires Bash on the backend, runs with errexit/pipefail, and
preserves verification metadata and output-failure detection. Configure its
default via `ExecutionToolset(..., verify_command="...")`. Git change detection
runs on the backend when Git is available; output truncation cannot count as a
passing verification result.
`process` supports `start`, `poll`, `input`, and `terminate`; IDs belong to one
open toolset. Polling returns cumulative bounded output. Handles and live
processes do not survive closing the toolset or restarting Harness. Completed
handles are evicted oldest-first after four times `max_processes` are retained.
Shell calls use independent shells; environment changes and `cd` do not carry
between separate shell calls. Use `cwd` for an explicit command directory or a
long-lived interactive process when shell state must persist.

| Backend | Selection and requirements | Lifecycle |
| --- | --- | --- |
| Local | `backend="local"`; POSIX `/bin/sh`, or native Windows with explicit `windows_shell` Bash executable | POSIX process groups or Windows Job Objects own descendants. POSIX PTY has a controlling terminal; Windows uses interactive pipes. |
| Docker | `backend="docker", docker_image="..."`; Docker daemon; image has `python3` and `/bin/sh` | Creates a uniquely named container, bind-mounts cwd at `/workspace`, and force-removes only that owned container at close. Network defaults to `none`; `docker_network="bridge"` enables it explicitly. |
| SSH | `backend="ssh", ssh_host="...", ssh_cwd="/absolute/path"`; OpenSSH, key authentication, POSIX remote Python 3.11+ | Every operation uses BatchMode SSH; remote supervisor enforces deadlines and process-group cleanup on stdin disconnect. No live external host was used in offline tests. |
| Singularity / Apptainer | `backend="singularity", singularity_image="..."`; set `singularity_binary="apptainer"` where appropriate; image has Python 3.11+ | Uses `exec --containall --cleanenv`, binds cwd at `/workspace`, and owns each execution transport. Tested with a fake transport, not a real Singularity installation. |
| Modal | `backend="modal"`; install `tools-execution[modal]`; SDK credentials configured normally; `modal_app` and `modal_image` select the app/image | Creates a named sandbox with a maximum lifetime, uploads the execution workers through the SDK, and terminates the owned sandbox on close. |
| Daytona | `backend="daytona"`; install `tools-execution[daytona]`; SDK credentials configured normally; optional `daytona_snapshot` | Creates an ephemeral named sandbox with TTL, uploads through the SDK, and waits for deletion on close. |
| Vercel Sandbox | `backend="vercel_sandbox"`; install `tools-execution[vercel]`; Vercel SDK credentials/project configured normally; optional `vercel_image` | Uses the Vercel 0.10 asynchronous API, creates a nonpersistent named sandbox with an execution limit, and destroys it on close. |

Cloud workspaces default to `/tmp/harness-workspace`; `cloud_cwd` overrides the
remote path. No host files are uploaded implicitly. Configure explicit files:

```python
config = ExecutionConfig(
    backend="modal",
    import_paths=("script.py", "input.bin"),
    export_paths=("results/output.bin",),
    export_directory="artifacts/execution",
)
async with ExecutionToolset(config, cwd=Path.cwd()) as execution:
    # Run commands through execution.tools.
    pass
print(execution.exported_files)
```

All imports are checked before creating resources or uploading anything.
Exports run after process cleanup and before sandbox destruction, preserving
binary contents under a unique local subdirectory without overwriting existing
files. Transfers accept explicit relative files, exclude `.harness`, reject
symlink components and traversal, and enforce per-file/total/count limits.
Export failure is observable even though the owned sandbox is still cleaned up.
Exported paths remain available via `execution.exported_files` after close.
Snapshot persistence and automatic whole-workspace synchronization are not
provided. `cloud_lifetime_seconds` bounds the provider resource lifetime;
commands also have independent backend-enforced deadlines. Missing optional
SDKs produce a dependency diagnostic before any requested command executes.
Cloud process controls and bounded logs live in a separate owned remote
directory; the same stdlib supervisor handles input, deadlines, and groups.

Remote setup errors fail closed. They never run the requested operation on the
host. Filesystem paths and explicit command cwd are checked on the selected
backend, including canonical symlink targets. Shell commands retain that
backend's OS permissions: a cwd check does not isolate arbitrary commands from
the rest of the local/SSH host. Docker exposes the selected workspace bind mount.
POSIX group cleanup handles ordinary descendants; deliberately detached process
sessions require stronger OS isolation. SSH network partitions cannot provide
immediate proof of cleanup, but each remote process has its own deadline.

Local POSIX subprocess/PTY and mocked transport tests run offline. The Docker
integration test runs only with an explicit image:

```sh
HARNESS_EXECUTION_TEST_IMAGE=python:3.12-slim uv run pytest packages/tools-execution/tests/test_docker_live.py
```

Cloud SDK tests bind calls against installed Modal 1.5.5, Daytona 0.211.2, and
Vercel 0.10.0 signatures while fake providers execute uploaded worker code in
temporary local fixture paths. They verify all file operations, shell and
verification, interactive processes, output caps, termination, and owned
cleanup. No live cloud service or external SSH host was used, so credential,
provider-image, and service behavior require deployment integration validation.
Native Windows local execution requires an explicit Git for Windows (or other
native Bash) executable; no shell selection falls back implicitly:

```toml
[execution]
backend = "local"
windows_shell = 'C:\Program Files\Git\bin\bash.exe'
```

The transport starts a gated Python process and assigns it to a Windows Job
Object with `KILL_ON_JOB_CLOSE` before releasing the selected executable or any
worker input. Descendants inherit the job; closing or cancelling the transport
closes the job. Assignment failure refuses execution. Windows pipes support
start/poll/input/terminate and independent deadlines; PTY requests fail with a
diagnostic because ConPTY is not implemented. Jobs manage ordinary process
descendants, not hostile code asking unrelated system services to spawn work.
SDK-only cloud clients do not need a POSIX host, but their workers require a
POSIX sandbox.

Windows contracts are covered by mocked Win32 calls and a full pipe/transport
test with a POSIX job stand-in on macOS. The native Windows test exercises actual
Job Objects, files, interactive input, timeout, background-child cleanup and
verification; it runs only on Windows with `HARNESS_WINDOWS_BASH` configured.
Native Windows acceptance depends on that CI job; this development host has not
executed Windows APIs.

The seven backend names match the pinned Hermes terminal inventory. Reference:
[terminal backend inventory at 939e45c](https://github.com/NousResearch/hermes-agent/blob/939e45c91d751fadd94dcd1b873ac3cb44846213/tools/terminal_tool.py).
