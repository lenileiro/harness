# tools-shell

Subprocess execution tool for Harness agents.

Tools:
- `shell(command, timeout=30)` — run a command in the session cwd, return stdout/stderr/exit code

Default approval policy: **prompt** (the agent must get user approval before each call).
