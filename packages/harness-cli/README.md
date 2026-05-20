# harness-cli

Typer + Rich command-line interface for Harness.

Installs a `harness` binary on the PATH:

```bash
harness run "your prompt"
harness chat                       # interactive REPL
harness sessions list
harness sessions show <id>
harness sessions resume <id>
harness providers list
harness tools list
```

Config lives at `~/.config/harness/config.toml`. Session storage defaults to `~/.local/state/harness/sessions.db`.
