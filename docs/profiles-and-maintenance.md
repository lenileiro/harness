# Profiles, services and maintenance

Create a named identity and use it for the entire CLI process:

```sh
harness profiles create work --workspace /absolute/path/to/workspace
harness --profile work setup --provider ollama --model YOUR_MODEL
harness auth set OPENAI_API_KEY --profile work
harness profiles persona work --text 'Prefer concise progress updates.'
harness --profile work chat
```

The credential command prompts without echoing the value. `auth status` lists only names; `auth remove NAME --profile work` removes a selected entry. Avoid secrets on command lines where shell history and process viewers can retain them. `profiles select work` persists a default profile; the explicit global `--profile` takes precedence.

Profiles default to `~/.harness/profiles/NAME`; `HARNESS_PROFILES_ROOT` overrides the catalog location. Each profile has its own config, credentials.env, state, skills, persona and default workspace. Activating it binds HARNESS_HOME/HARNESS_CONFIG/CODEX_HOME for the whole process, removes inherited key/token/secret/password variables and restores the previous environment and working directory on exit. Reserved runtime variables cannot be stored as credentials. Credential writes are private, atomic and serialized; symlink files are rejected.

The global `--profiles-root PATH` flag selects a catalog explicitly; `--home PATH`
selects state for the default identity. Named profiles keep their own state.
Service launch arguments persist these paths so boot does not depend on terminal
environment variables.

Common AWS, Google Cloud, Azure, Modal and Hugging Face credential/config locations default inside the profile. AWS instance-metadata credentials are disabled by default. Explicit credential-file/config variables entered for that profile can select an external account file deliberately. This prevents common accidental account borrowing; it is not a sandbox against arbitrary trusted plugins or SDK-specific global caches. Local shell/computer tools retain the operating-system permissions of the launching user.

SOUL.md is bounded local persona text. Remote users do not inherit the profile's local persona or managed credentials automatically. Run a separate process per profile; one multi-channel process does not switch process-global identity during a message.

## Background services

Install an inspectable command, then start its supervisor:

```sh
harness --profile work service install channels --cwd /absolute/path/to/workspace -- harness channel run-all
harness --profile work service start channels
harness --profile work service status channels
harness --profile work service logs channels --lines 100
harness --profile work service restart channels
harness --profile work service stop channels
```

Install records argv without a shell, working directory and restart budget. Commands should reference environment variable names, not inline secrets. State records process creation identity to prevent PID reuse from stopping another process. Locks prevent duplicate supervisors. The supervisor owns descendants, records crashes and bounds restart attempts; stopping it terminates its owned child group. Services inherit the selected profile when started.

Installation writes a platform boot manifest for review: systemd user service on Linux, LaunchAgent on macOS, or least-privilege scheduled task on Windows. `service enable-boot NAME` and `disable-boot NAME` explicitly register/remove it. Merely installing a service does not alter OS boot settings. Native boot registration must be verified on the target OS.

## Backup and restore

```sh
harness --profile work maintenance backup /safe/location/work-backup.tar.gz
harness maintenance restore /safe/location/work-backup.tar.gz --destination /new/restore-directory
```

Backup includes the selected identity plus its external current-workspace state and XDG config/session database where applicable. SQLite databases use consistent backup snapshots, including committed WAL contents. Known credential files and auth directories, including managed Weixin and Photon accounts, are omitted by default; `--include-credentials` explicitly includes them. Saved conversations and channel routing state remain private even when credential files are omitted. Repository metadata, dependency caches, symlinks and special files are skipped. Archives are bounded to 10,000 files and 1 GiB and contain a versioned size/hash manifest.

Restore checks names, duplicate paths, file types, hashes, sizes and aggregate limits before publishing a staged directory. The destination must be new. Restoring an archive does not activate a profile or rewrite absolute paths in saved configuration; inspect and rebind its workspace/profile paths before selecting it on another machine.

`maintenance update --check --checkout PATH` fetches and reports upstream status for a clean source checkout. Omitting `--check` performs a fast-forward update and locked dependency sync. Local changes must be saved first. This implementation has not run self-update, boot registration or a deployment.
