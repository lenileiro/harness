# Persistence

Harness uses SQLite by default for local development and single-user workflows.

Why:
- zero-setup local storage
- easy portability across laptops and CI smoke runs
- simple backup and inspection

Tradeoffs:
- limited concurrent writes compared with a client/server database
- shared or team deployments may eventually want Postgres or another networked store
