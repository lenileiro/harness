# OpenClaw migration

Migration runs offline against an explicitly selected directory. It creates a **new, inactive Harness profile** with its own workspace, database, skills, persona and optional credentials. Existing profiles and the OpenClaw source are never overwritten.

```sh
harness migrate openclaw inspect --source /path/to/openclaw --profile imported --output migration-plan.json
# Read migration-plan.json, especially settings, warnings and available_credentials.
harness migrate openclaw apply --plan migration-plan.json
harness --profile imported doctor
```

An external configured workspace requires `--workspace /explicit/path`. Select another configured agent with `--agent AGENT_ID`. The importer supports both `agents.entries` and older `agents.list` configuration layouts. A config using `$include` must first be exported as a flattened config; migration does not follow config includes or execute secret resolvers.

The plan contains file hashes, paths, translated settings, credential identifiers, and warnings. It contains no credential values or memory bodies. Apply re-reads the source and rejects a stale or modified plan. Stop the source agent while reviewing/applying if its SQLite database keeps changing. The final profile is published after staged file validation and database writes finish; failures remove staging files.

## What is translated

- JSON5 `agents.defaults.model` or selected-agent model to Harness provider/model defaults for OpenAI, Anthropic, OpenRouter and Ollama. Safe provider `baseUrl` settings are translated. Other provider kinds, aliases and fallback chains are reported for manual configuration.
- Workspace `USER.md`, `MEMORY.md` and `memory/**/*.md` to searchable SQLite memory scoped to the **new workspace**, with originals kept under `imports/openclaw/workspace`. Large files become bounded memory chunks. `SOUL.md` becomes the profile persona; `DREAMS.md` is retained as an archive.
- Managed and workspace skill packages to portable Agent Skills. Workspace packages take precedence. Ordinary display metadata can be converted to strings. Skills requiring OpenClaw command dispatch, install/runtime gates, secret injection, or disabled model invocation are archived for adaptation. They are not silently enabled. Dot files, including skill `.env` files, are excluded. No skill scripts or installers execute during import.
- Explicitly selected **static API keys** from source `.env`, config env/provider keys, current shared/per-agent SQLite stores, or legacy `auth-profiles.json` when no current database exists. Current SQLite stores always suppress legacy JSON fallback.

Channels, outbound integration credentials, cron jobs, transcripts, tool policies, sandbox settings, OAuth sessions, personal model-account records, and file/exec/keychain secret providers are not imported. Review the warnings and configure Harness policy before using the new profile. Unknown configuration sections are reported without copying their raw values.

## Selecting credentials

Inspect without credentials first. `available_credentials` lists safe source identifiers, credential kinds and whether each can be imported. Re-run inspect into a **new plan file**, selecting each required key:

```sh
harness migrate openclaw inspect --source /path/to/openclaw --profile imported --output selected-plan.json \
  --credential provider:openai=OPENAI_API_KEY
# Or a static per-agent auth profile:
# --credential agent:auth:openai:work=OPENAI_API_KEY
harness migrate openclaw apply --plan selected-plan.json
```

Other identifiers include `dotenv:OPENAI_API_KEY`, `config-env:OPENAI_API_KEY`, `shared:auth:PROFILE_ID` and `legacy:auth:PROFILE_ID`. Names are identifiers, never credential values. Only selected values enter the private `credentials.env`; neither config nor migration plan contains them. Parent-process environment variables are not used for credential resolution. OAuth/token records and unresolved references remain nonportable instead of being mislabeled as API keys.

Source reads pin directory components, reject symlinks/special files, and enforce limits: 4 MiB per ordinary file, 64 MiB per credential database/WAL file, 128 MiB overall, 2,000 files/directories and 16 nested skill/memory directories. `SOUL.md` is limited to 64,000 bytes. Migration currently requires POSIX no-follow directory-descriptor support. Profile directories are private and imported files have owner-only permissions.

## Source format references

The parser follows [OpenClaw configuration](https://docs.openclaw.ai/gateway/configuration), [agent model settings](https://docs.openclaw.ai/gateway/config-agents/models), [agent entries](https://docs.openclaw.ai/gateway/config-agents/entries-and-multi-agent), [memory](https://docs.openclaw.ai/concepts/memory), and [skills](https://docs.openclaw.ai/tools/skills). Current SQLite credential rows are grounded in [OpenClaw's auth persistence implementation](https://github.com/openclaw/openclaw/blob/main/src/agents/auth-profiles/sqlite.ts): `auth_profile_store(store_key='primary', store_json)` for agents; `config_machine_state(state_key='authProfiles.store', value_json)` for the shared store. Only those rows are read, including committed WAL data in private snapshots.
