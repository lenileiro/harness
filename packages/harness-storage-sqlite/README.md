# harness-storage-sqlite

SQLite-backed `Storage` implementation for Harness, using [aiosqlite](https://aiosqlite.omnilib.dev/).

Sessions are persisted across process restarts. Default location: `~/.local/state/harness/sessions.db`.
