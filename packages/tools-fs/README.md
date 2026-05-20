# tools-fs

Filesystem tools for Harness agents. All operations are scoped to the session's working directory and respect the configured approval policy.

Tools:
- `read_file(path)` — read text file
- `write_file(path, content)` — create or overwrite
- `edit_file(path, old, new)` — exact-string replacement
- `list_dir(path)` — list entries
- `glob(pattern)` — match files by glob
