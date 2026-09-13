"""Offline, reviewable OpenClaw import into a new, inactive Harness profile.

Only known settings are translated. Source secrets are never part of a plan;
explicit credential selections are resolved again while applying that plan.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import json5
import tomlkit
import yaml
from dotenv import dotenv_values, set_key
from pydantic import BaseModel, ConfigDict, Field

from harness.cli.config import load_config
from harness.cli.profiles import Profile, profile_root, profiles_root
from harness.core.memory import MemoryEntry, MemoryScope
from harness.core.skills import Skill, validate_skill_name
from harness.storage.sqlite import SQLiteStorage

MAX_FILE = 4 * 1024 * 1024
MAX_DATABASE = 64 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_FILES = 2000
_RESERVED = {
    "HOME",
    "PATH",
    "USERPROFILE",
    "HARNESS_HOME",
    "HARNESS_PROFILE",
    "HARNESS_CONFIG",
    "HARNESS_PROFILES_ROOT",
    "CODEX_HOME",
}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
# OpenClaw names the subscription-backed Claude Code CLI path `claude-code`;
# Harness calls that provider `claude` (the `anthropic` provider is API-key only).
_PROVIDER_ALIASES = {"claude-code": "claude", "claudecode": "claude"}


class MigrationError(ValueError):
    """A safe, content-free diagnostic suitable for command-line output."""


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Literal["source", "workspace"]
    path: str
    sha256: str
    size: int


class CredentialInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    kind: str
    importable: bool


class MigrationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    source: str
    profile: str
    agent: str = "main"
    workspace: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    files: list[SourceFile] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    available_credentials: list[CredentialInfo] = Field(default_factory=list)
    installed_skills: list[str] = Field(default_factory=list)
    archived_skills: list[str] = Field(default_factory=list)
    memory_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _parts(relative: str) -> tuple[str, ...]:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in {"..", "."} for p in parts):
        raise MigrationError("source paths must remain inside the selected directory")
    return parts


def _open_root(root: Path) -> int:
    # The user explicitly selects the root. Pin it and every descendant with
    # dirfd opens; resolving a source-controlled symlink would defeat the bound.
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise MigrationError("safe migration requires no-follow directory descriptor support")
    return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _open_child(root_fd: int, relative: str, *, directory: bool = False) -> int:
    parts = _parts(relative)
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = next_fd
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        return os.open(parts[-1], flags, dir_fd=parent)
    finally:
        os.close(parent)


@dataclass
class _Snapshot:
    roots: dict[str, Path]
    files: dict[tuple[str, str], bytes] = field(default_factory=dict)
    total: int = 0
    directories: int = 0
    workspace_relative: str | None = None

    def root_fd(self, root: str) -> int:
        if root == "workspace" and self.workspace_relative is not None:
            parent = _open_root(self.roots["source"])
            try:
                return _open_child(parent, self.workspace_relative, directory=True)
            finally:
                os.close(parent)
        return _open_root(self.roots[root])

    def read(self, root: str, relative: str, *, limit: int = MAX_FILE) -> bytes | None:
        key = (root, relative)
        if key in self.files:
            return self.files[key]
        root_fd = self.root_fd(root)
        try:
            try:
                fd = _open_child(root_fd, relative)
            except FileNotFoundError:
                return None
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                    raise MigrationError("source contains a special or oversized file")
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise MigrationError("source file exceeds the migration size limit")
            self.total += len(data)
            if self.total > MAX_TOTAL or len(self.files) >= MAX_FILES:
                raise MigrationError("source exceeds the migration file count or total size limit")
            self.files[key] = data
            return data
        finally:
            os.close(root_fd)

    def children(self, root: str, relative: str) -> list[tuple[str, bool]]:
        self.directories += 1
        if self.directories > MAX_FILES:
            raise MigrationError("source contains too many directories")
        root_fd = self.root_fd(root)
        try:
            try:
                fd = _open_child(root_fd, relative, directory=True)
            except FileNotFoundError:
                return []
            try:
                result: list[tuple[str, bool]] = []
                with os.scandir(fd) as entries:
                    for entry in entries:
                        if len(result) >= MAX_FILES:
                            raise MigrationError("source directory contains too many entries")
                        info = entry.stat(follow_symlinks=False)
                        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                            raise MigrationError("source contains a symlink or special file")
                        result.append((entry.name, stat.S_ISDIR(info.st_mode)))
                return sorted(result)
            finally:
                os.close(fd)
        finally:
            os.close(root_fd)

    def tree(self, root: str, relative: str, *, depth: int = 0) -> dict[str, bytes]:
        if depth > 16:
            raise MigrationError("source directory nesting exceeds the migration limit")
        result: dict[str, bytes] = {}
        for name, directory in self.children(root, relative):
            # Dot files are not portable skill assets; in particular .env and
            # source-control metadata must not become implicit credential copies.
            if name.startswith("."):
                continue
            path = f"{relative}/{name}"
            if directory:
                result.update(self.tree(root, path, depth=depth + 1))
            else:
                data = self.read(root, path)
                if data is not None:
                    result[path] = data
        return result

    def manifest(self) -> list[SourceFile]:
        return [
            SourceFile(
                root=cast(Literal["source", "workspace"], root),
                path=path,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            for (root, path), data in sorted(self.files.items())
        ]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json(data: bytes, *, json_five: bool = False) -> dict[str, Any]:
    try:
        result = (
            json5.loads(data.decode(), allow_duplicate_keys=False)
            if json_five
            else json.loads(data)
        )
    except (ValueError, UnicodeError, RecursionError):
        raise MigrationError("source JSON/JSON5 is invalid; inspect its syntax locally") from None
    if not isinstance(result, dict):
        raise MigrationError("source JSON/JSON5 must contain an object")
    return result


def _database_store(snapshot: _Snapshot, relative: str, *, shared: bool) -> dict[str, Any] | None:
    data = snapshot.read("source", relative, limit=MAX_DATABASE)
    if data is None:
        return None
    wal = snapshot.read("source", relative + "-wal", limit=MAX_DATABASE)
    # Query a private bounded copy, never the live source (SQLite may create
    # shared-memory files even for a read-only WAL database).
    with tempfile.TemporaryDirectory(prefix="harness-openclaw-db-") as temporary:
        path = Path(temporary) / "snapshot.sqlite"
        path.write_bytes(data)
        path.chmod(0o600)
        if wal is not None:
            Path(str(path) + "-wal").write_bytes(wal)
        try:
            with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
                db.execute("PRAGMA trusted_schema=OFF")
                db.execute("PRAGMA query_only=ON")
                table = "config_machine_state" if shared else "auth_profile_store"
                found = db.execute(
                    "SELECT type FROM sqlite_master WHERE name=?", (table,)
                ).fetchone()
                if not found:
                    return {}
                if found[0] != "table":
                    raise MigrationError("credential database contains an unsupported schema")
                query = (
                    "SELECT value_json FROM config_machine_state WHERE state_key='authProfiles.store'"
                    if shared
                    else "SELECT store_json FROM auth_profile_store WHERE store_key='primary'"
                )
                row = db.execute(query).fetchone()
                if row is None:
                    return {}
                if not isinstance(row[0], str) or len(row[0]) > MAX_FILE:
                    raise MigrationError("credential database payload is invalid or oversized")
                return _json(row[0].encode())
        except sqlite3.Error:
            raise MigrationError(
                "credential database could not be read; stop OpenClaw and inspect its schema"
            ) from None


def _credential_value(value: Any, env: dict[str, str]) -> str | None:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([A-Z_][A-Z0-9_]*)\}", value)
        if match:
            return env.get(match[1])
        if "${" in value:
            return None
        return value if value.strip() else None
    if (
        isinstance(value, dict)
        and value.get("source") == "env"
        and value.get("provider", "default") == "default"
    ):
        return env.get(str(value.get("id")))
    return None


def _valid_variable(value: str) -> bool:
    return bool(_ENV_NAME.fullmatch(value)) and value not in _RESERVED


def _skill_document(data: bytes, name: str) -> tuple[bytes, bool]:
    try:
        lines = data.decode().splitlines()
        if not lines or lines[0].strip() != "---":
            return data, False
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        front = yaml.safe_load("\n".join(lines[1:end]))
        if not isinstance(front, dict) or front.get("name") != name:
            return data, False
        metadata = front.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json5.loads(metadata)
        if not isinstance(metadata, dict):
            return data, False
        gating = _mapping(metadata.get("openclaw"))
        # These fields carry runtime behavior that Harness does not emulate.
        if (
            any(key in front for key in ("command-dispatch", "command-tool", "command-arg-mode"))
            or front.get("disable-model-invocation")
            or any(key in gating for key in ("requires", "os", "install", "primaryEnv", "skillKey"))
        ):
            return data, False
        if any(
            key
            not in {
                "name",
                "description",
                "metadata",
                "license",
                "compatibility",
                "allowed-tools",
                "user-invocable",
                "disable-model-invocation",
            }
            for key in front
        ):
            return data, False
        front["metadata"] = {
            str(key): value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            for key, value in metadata.items()
        }
        result = (
            "---\n"
            + yaml.safe_dump(front, sort_keys=False)
            + "---\n"
            + "\n".join(lines[end + 1 :])
            + "\n"
        ).encode()
        with tempfile.TemporaryDirectory(prefix="harness-openclaw-skill-") as temporary:
            directory = Path(temporary) / name
            directory.mkdir()
            (directory / "SKILL.md").write_bytes(result)
            Skill.load(directory)
        return result, True
    except (ValueError, UnicodeError, yaml.YAMLError, StopIteration, RecursionError):
        return data, False


@dataclass
class _Prepared:
    plan: MigrationPlan
    snapshot: _Snapshot
    copies: dict[str, bytes]
    memories: dict[str, str]
    secrets: dict[str, str]


def _prepare(
    source: Path,
    profile: str,
    *,
    agent: str = "main",
    workspace: Path | None = None,
    credentials: dict[str, str] | None = None,
) -> _Prepared:
    Profile(name=profile, workspace=Path.cwd())
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", agent):
        raise MigrationError("agent identifier contains unsupported characters")
    source = Path(os.path.abspath(source.expanduser()))
    if source.is_symlink():
        raise MigrationError("selected source root must not be a symlink")
    selected = credentials or {}
    if len(selected.values()) != len(set(selected.values())) or any(
        not _valid_variable(value) for value in selected.values()
    ):
        raise MigrationError(
            "credential destination variables must be unique, valid and non-reserved"
        )
    snapshot = _Snapshot({"source": source})
    config_data = snapshot.read("source", "openclaw.json")
    config = _json(config_data, json_five=True) if config_data else {}
    if "$include" in config:
        raise MigrationError(
            "config includes require a flattened OpenClaw config export before migration"
        )
    agents = _mapping(config.get("agents"))
    defaults = _mapping(agents.get("defaults"))
    entries = _mapping(agents.get("entries"))
    legacy_entries = agents.get("list", [])
    if not entries and isinstance(legacy_entries, list):
        entries = {str(item.get("id")): item for item in legacy_entries if isinstance(item, dict)}
    if entries and agent not in entries:
        raise MigrationError("selected agent is not present in the OpenClaw configuration")
    entry = _mapping(entries.get(agent))
    warnings: list[str] = []
    configured_workspace = entry.get("workspace", defaults.get("workspace"))
    workspace_root = workspace.expanduser().absolute() if workspace else source / "workspace"
    if not workspace and isinstance(configured_workspace, str):
        candidate = Path(configured_workspace).expanduser()
        candidate = Path(
            os.path.abspath(candidate if candidate.is_absolute() else source / candidate)
        )
        if not candidate.absolute().is_relative_to(source):
            raise MigrationError(
                "configured workspace is outside --source; select it explicitly with --workspace"
            )
        workspace_root = candidate.absolute()
    if workspace_root.is_symlink():
        raise MigrationError("selected workspace root must not be a symlink")
    snapshot.roots["workspace"] = workspace_root
    if not workspace and workspace_root != source:
        snapshot.workspace_relative = workspace_root.relative_to(source).as_posix()
    plan = MigrationPlan(
        source=str(source),
        profile=profile,
        agent=agent,
        workspace=str(workspace.expanduser().absolute()) if workspace else None,
        credentials=selected,
    )
    for key in sorted(config):
        if key not in {"agents", "models", "env", "auth", "skills", "meta", "wizard"}:
            warnings.append(f"Configuration section {key!r} is not imported.")
    warnings.append(
        "Tool permissions, integrations, scheduling, sandbox settings and source transcripts are not imported; review Harness policy before using the new profile."
    )
    model = entry.get("model", defaults.get("model"))
    model_record = _mapping(model)
    if model_record.get("fallbacks"):
        warnings.append("Model fallback chains require manual configuration.")
    primary = model_record.get("primary") if model_record else model
    settings: dict[str, Any] = {}
    providers = _mapping(_mapping(config.get("models")).get("providers"))
    if isinstance(primary, str) and _MODEL.fullmatch(primary) and "/" in primary:
        source_provider, model_name = primary.split("/", 1)
        provider = _PROVIDER_ALIASES.get(source_provider, source_provider)
        if provider in {"openai", "anthropic", "openrouter", "ollama", "claude"}:
            settings["default"] = {"provider": provider, "model": model_name}
            provider_config = _mapping(providers.get(source_provider))
            base_url = provider_config.get("baseUrl")
            # The Claude provider drives a local CLI, so a source base URL has
            # nothing to configure on this side.
            if isinstance(base_url, str) and provider != "claude":
                parsed = urlsplit(base_url)
                if (
                    parsed.scheme in {"http", "https"}
                    and parsed.hostname
                    and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
                ):
                    settings["provider"] = {provider: {"base_url": base_url}}
                else:
                    warnings.append(
                        "Provider URL was not imported because it contains unsupported credentials or URL components."
                    )
        else:
            warnings.append(
                f"Provider {provider!r} needs a Harness provider configuration; its model default was not imported."
            )
    elif primary is not None:
        warnings.append(
            "Model default is an alias or unsupported expression; select provider/model with harness setup."
        )
    plan.settings = settings
    # Only source-local environment files are considered, never os.environ or
    # shellEnv / file / exec / OS-keychain credential providers.
    env_values: dict[str, str] = {}
    available: dict[str, tuple[str, str | None]] = {}
    env_data = snapshot.read("source", ".env")
    if env_data:
        try:
            parsed_env = dotenv_values(stream=io.StringIO(env_data.decode()), interpolate=False)
        except UnicodeError:
            raise MigrationError("source .env must be UTF-8") from None
        for key, value in parsed_env.items():
            if isinstance(value, str) and _valid_variable(key):
                env_values[key] = value
                available[f"dotenv:{key}"] = ("api_key", value)
    config_env = _mapping(config.get("env"))
    config_vars = {
        **{key: value for key, value in config_env.items() if isinstance(value, str)},
        **_mapping(config_env.get("vars")),
    }
    for key, value in config_vars.items():
        if isinstance(value, str) and _valid_variable(key):
            env_values.setdefault(key, value)
            available[f"config-env:{key}"] = ("api_key", value)
    for key, value in providers.items():
        candidate = _mapping(value).get("apiKey")
        if candidate is not None:
            available[f"provider:{key}"] = ("api_key", _credential_value(candidate, env_values))
    agent_dir = entry.get("agentDir")
    agent_relative = f"agents/{agent}/agent"
    if agent_dir:
        candidate = Path(str(agent_dir)).expanduser()
        candidate = Path(
            os.path.abspath(candidate if candidate.is_absolute() else source / candidate)
        )
        if not candidate.absolute().is_relative_to(source):
            warnings.append(
                "External agentDir credentials are not inspected; use a source-local exported copy."
            )
            agent_relative = ""
        else:
            agent_relative = candidate.absolute().relative_to(source).as_posix()
    shared = _database_store(snapshot, "state/openclaw.sqlite", shared=True)
    local = (
        _database_store(snapshot, f"{agent_relative}/openclaw-agent.sqlite", shared=False)
        if agent_relative
        else None
    )
    stores: list[tuple[str, dict[str, Any]]] = []
    if shared is not None:
        stores.append(("shared", shared))
    if local is not None:
        stores.append(("agent", local))
    if shared is None and local is None and agent_relative:
        legacy = snapshot.read("source", f"{agent_relative}/auth-profiles.json")
        if legacy:
            stores.append(("legacy", _json(legacy)))
            warnings.append(
                "Legacy auth-profiles.json was inspected because no current SQLite store exists; OAuth credentials are not portable."
            )
    elif shared is not None or local is not None:
        warnings.append(
            "Current SQLite credentials take precedence; retired JSON auth files are not inspected or imported."
        )
    for location, store in stores:
        for key, value in _mapping(store.get("profiles")).items():
            record = _mapping(value)
            kind = str(record.get("type", "unknown"))
            secret = (
                _credential_value(record.get("keyRef", record.get("key")), env_values)
                if kind == "api_key"
                else None
            )
            available[f"{location}:auth:{key}"] = (
                kind if kind in {"api_key", "token", "oauth"} else "unsupported",
                secret,
            )
    plan.available_credentials = [
        CredentialInfo(source=key, kind=kind, importable=secret is not None)
        for key, (kind, secret) in sorted(available.items())
    ]
    secrets: dict[str, str] = {}
    for key, variable in selected.items():
        record = available.get(key)
        if not record or record[1] is None:
            raise MigrationError(
                "selected credential is missing, unresolved, OAuth, or otherwise nonportable; inspect available_credentials"
            )
        if "\x00" in record[1]:
            raise MigrationError("selected credential contains unsupported data")
        secrets[variable] = record[1]
    copies: dict[str, bytes] = {}
    memories: dict[str, str] = {}
    if workspace_root.exists():
        for name in ("USER.md", "MEMORY.md", "SOUL.md", "DREAMS.md"):
            data = snapshot.read("workspace", name, limit=64000 if name == "SOUL.md" else MAX_FILE)
            if data is None:
                continue
            try:
                decoded = data.decode()
            except UnicodeError:
                raise MigrationError("workspace Markdown files must be UTF-8") from None
            copies[f"imports/openclaw/workspace/{name}"] = data
            if name in {"USER.md", "MEMORY.md"} and decoded.strip():
                memories[name] = decoded
            elif name == "SOUL.md":
                copies["SOUL.md"] = data
        for relative, data in snapshot.tree("workspace", "memory").items():
            if relative.endswith(".md"):
                try:
                    memories[relative] = data.decode()
                except UnicodeError:
                    raise MigrationError("memory Markdown files must be UTF-8") from None
                copies[f"imports/openclaw/workspace/{relative}"] = data
    else:
        warnings.append(
            "Selected workspace does not exist; no workspace memory or skills were imported."
        )
    # Workspace skills override managed skills with the same name. Incompatible
    # OpenClaw dispatch/gating remains archived, never silently made executable.
    skill_sources: dict[str, str] = {}
    for root in ("source", "workspace"):
        if not snapshot.roots[root].exists():
            continue
        for name, directory in snapshot.children(root, "skills"):
            if name.startswith(".") or not directory:
                continue
            try:
                validate_skill_name(name)
            except ValueError:
                warnings.append("A skill with a nonportable name was skipped.")
                continue
            skill_sources[name] = root
    skill_entries = _mapping(_mapping(config.get("skills")).get("entries"))
    selected_skills = entry.get("skills")
    for name, root in sorted(skill_sources.items()):
        files = snapshot.tree(root, f"skills/{name}")
        document = files.get(f"skills/{name}/SKILL.md")
        if document is None:
            warnings.append(f"Skill {name!r} has no SKILL.md and was skipped.")
            continue
        converted, compatible = _skill_document(document, name)
        skill_setting = _mapping(skill_entries.get(name))
        enabled = skill_setting.get("enabled") is not False and not any(
            key in skill_setting for key in ("env", "apiKey", "config")
        )
        if isinstance(selected_skills, list):
            enabled = enabled and name in selected_skills
        for path, data in files.items():
            copies[f"imports/openclaw/{root}/{path}"] = data
            if compatible and enabled:
                copies[path] = converted if path == f"skills/{name}/SKILL.md" else data
        if compatible and enabled:
            plan.installed_skills.append(name)
        else:
            plan.archived_skills.append(name)
            warnings.append(
                f"Skill {name!r} was archived: disabled, invalid, or dependent on OpenClaw runtime behavior."
            )
    plan.files = snapshot.manifest()
    plan.memory_files = sorted(memories)
    plan.warnings = warnings
    return _Prepared(plan, snapshot, copies, memories, secrets)


def inspect_openclaw(
    source: Path,
    profile: str,
    *,
    agent: str = "main",
    workspace: Path | None = None,
    credentials: dict[str, str] | None = None,
) -> MigrationPlan:
    try:
        return _prepare(
            source, profile, agent=agent, workspace=workspace, credentials=credentials
        ).plan
    except MigrationError:
        raise
    except (OSError, ValueError):
        raise MigrationError(
            "unable to inspect source safely; check selected directories, file types and configuration"
        ) from None


def _write(root: Path, relative: str, data: bytes) -> None:
    path = root.joinpath(*_parts(relative))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _publish(staging: Path, target: Path) -> None:
    for path in staging.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    target.mkdir(mode=0o700, exist_ok=False)
    claim = target.stat(follow_symlinks=False)
    try:
        staging.rename(target)
    except BaseException:
        current = target.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (claim.st_dev, claim.st_ino):
            target.rmdir()
        raise


def _cleanup(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)


async def apply_openclaw(plan: MigrationPlan) -> Profile:
    """Re-read reviewed source and publish a complete new profile; never activate it."""
    target = profile_root(plan.profile)
    if target.exists() or target.is_symlink():
        raise MigrationError(
            "target profile already exists; choose a new profile name and inspect again"
        )
    try:
        prepared = _prepare(
            Path(plan.source),
            plan.profile,
            agent=plan.agent,
            workspace=Path(plan.workspace) if plan.workspace else None,
            credentials=plan.credentials,
        )
    except MigrationError:
        raise
    except (OSError, ValueError):
        raise MigrationError(
            "source can no longer be inspected safely; create a new plan"
        ) from None
    if prepared.plan != plan:
        raise MigrationError(
            "migration plan is stale or was modified; inspect again and review the new plan"
        )
    root = profiles_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    profile = Profile(
        name=plan.profile,
        workspace=target / "workspace",
        description=f"Imported from OpenClaw agent {plan.agent}",
    )
    staging = Path(tempfile.mkdtemp(prefix=".openclaw-import-", dir=root))
    try:
        (staging / "workspace").mkdir(mode=0o700)
        _write(staging, "profile.json", profile.model_dump_json(indent=2).encode())
        _write(staging, "config.toml", tomlkit.dumps(plan.settings).encode())
        load_config(staging / "config.toml")
        _write(staging, "imports/openclaw/plan.json", plan.model_dump_json(indent=2).encode())
        for relative, data in prepared.copies.items():
            _write(staging, relative, data)
        for name in plan.installed_skills:
            Skill.load(staging / "skills" / name)
        if prepared.secrets:
            _write(staging, "credentials.env", b"")
            for variable, secret in prepared.secrets.items():
                set_key(staging / "credentials.env", variable, secret, quote_mode="always")
            (staging / "credentials.env").chmod(0o600)
        storage = SQLiteStorage(path=staging / "state/sessions.db")
        scope = MemoryScope(workspace=str(profile.workspace))
        try:
            for relative, content in prepared.memories.items():
                # Preserve the original archive and create searchable, scoped
                # memory chunks without one giant injected context entry.
                for offset in range(0, len(content), 4000):
                    await storage.save_scoped_memory(
                        MemoryEntry(
                            kind="user_preference" if relative == "USER.md" else "project_context",
                            text=f"Imported OpenClaw memory ({relative}, part {offset // 4000 + 1}):\n{content[offset : offset + 4000]}",
                        ),
                        scope=scope,
                    )
        finally:
            await storage.close()
        # Publish only after all writes, validation and SQLite closure succeed.
        _publish(staging, target)
        return profile
    finally:
        _cleanup(staging)
