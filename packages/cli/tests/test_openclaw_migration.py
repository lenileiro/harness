from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
import typer
from dotenv import dotenv_values
from typer.testing import CliRunner

from harness.cli.config import load_config
from harness.cli.migration_commands import migration_app
from harness.cli.openclaw_migration import MigrationError, apply_openclaw, inspect_openclaw
from harness.cli.profiles import activate_profile, profile_root
from harness.core.memory import MemoryScope
from harness.core.skills import SkillLibrary, default_skill_paths
from harness.storage.sqlite import SQLiteStorage, default_db_path


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    root = tmp_path / "openclaw"
    (root / "workspace/memory").mkdir(parents=True)
    (root / "workspace/USER.md").write_text("Prefer short factual replies.")
    (root / "workspace/MEMORY.md").write_text("The project uses SQLite.")
    (root / "workspace/memory/2026-09-01.md").write_text("Observed a successful offline run.")
    (root / "workspace/SOUL.md").write_text("Work carefully and cite uncertainty.")
    (root / ".env").write_text(
        "OPENAI_API_KEY='selected-secret'\nUNRELATED_TOKEN='unselected-secret'\n"
    )
    (root / "openclaw.json").write_text("""{
      // Real OpenClaw JSON5: comments, bare keys and trailing commas.
      agents: {defaults: {model: {primary: "openai/example-model", fallbacks: ["anthropic/other"]}}},
      models: {providers: {openai: {baseUrl: "https://api.openai.com/v1", apiKey: "${OPENAI_API_KEY}"}}},
      channels: {slack: {botToken: "do-not-copy-channel-secret"}},
    }""")
    return root


def skill(source: Path, name: str, front: str = "", body: str = "Use this guide."):
    root = source / "skills" / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: An imported guide.\n{front}---\n{body}\n"
    )
    (root / "reference.txt").write_text("Portable reference material.")
    (root / ".env").write_text("PACKAGE_SECRET=never-copy-me")
    return root


def auth_database(source: Path, profiles: dict, *, shared=False, wal=False):
    path = source / (
        "state/openclaw.sqlite" if shared else "agents/main/agent/openclaw-agent.sqlite"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    if wal:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=0")
    if shared:
        db.execute("CREATE TABLE config_machine_state(state_key TEXT PRIMARY KEY,value_json TEXT)")
        db.execute(
            "INSERT INTO config_machine_state VALUES (?,?)",
            ("authProfiles.store", json.dumps({"version": 1, "profiles": profiles})),
        )
        # Personal credential rows must never be enumerated by this importer.
        db.execute(
            "INSERT INTO config_machine_state VALUES (?,?)",
            ("model-account:private", json.dumps({"key": "private-unrelated-secret"})),
        )
    else:
        db.execute("CREATE TABLE auth_profile_store(store_key TEXT PRIMARY KEY,store_json TEXT)")
        db.execute(
            "INSERT INTO auth_profile_store VALUES (?,?)",
            ("primary", json.dumps({"version": 1, "profiles": profiles})),
        )
    db.commit()
    return db


@pytest.mark.asyncio
async def test_reviewed_import_is_usable_and_isolated_without_implicit_secrets(source):
    skill(source, "portable-guide", 'metadata: {openclaw: {emoji: "book"}}\n')
    plan = inspect_openclaw(source, "imported")
    serialized = plan.model_dump_json()
    assert "selected-secret" not in serialized and "unselected-secret" not in serialized
    assert "do-not-copy-channel-secret" not in serialized
    assert plan.installed_skills == ["portable-guide"]
    profile = await apply_openclaw(plan)
    root = profile_root(profile.name)
    assert not (root / "credentials.env").exists()
    assert not (root.parent / "active.json").exists()
    assert not list(root.rglob(".env"))
    assert load_config(root / "config.toml").default_model == "example-model"
    with activate_profile(profile.name):
        storage = SQLiteStorage(path=default_db_path())
        try:
            entries = await storage.list_scoped_memory(
                scope=MemoryScope(workspace=str(profile.workspace))
            )
            assert len(entries) == 3
            assert any("SQLite" in entry.text for entry in entries)
            assert await storage.list_scoped_memory(scope=MemoryScope(workspace=str(source))) == []
        finally:
            await storage.close()
        library = SkillLibrary.load(default_skill_paths(profile.workspace))
        assert "portable-guide" in library.skills and not library.errors
    assert (root / "SOUL.md").read_text().startswith("Work carefully")
    assert (root / "config.toml").stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700
    assert "selected-secret" not in "\n".join(
        path.read_text()
        for path in root.rglob("*")
        if path.is_file() and path.suffix not in {".db"}
    )


@pytest.mark.asyncio
async def test_only_explicit_credential_selection_resolves_source_env(source, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-win")
    plan = inspect_openclaw(source, "keys", credentials={"provider:openai": "OPENAI_API_KEY"})
    await apply_openclaw(plan)
    root = profile_root("keys")
    assert dict(dotenv_values(root / "credentials.env")) == {"OPENAI_API_KEY": "selected-secret"}
    assert "selected-secret" not in (root / "imports/openclaw/plan.json").read_text()
    assert "unselected-secret" not in (root / "credentials.env").read_text()


@pytest.mark.asyncio
async def test_sqlite_current_store_with_wal_and_private_shared_records(source):
    local = auth_database(
        source,
        {"openai:work": {"type": "api_key", "provider": "openai", "key": "wal-secret"}},
        wal=True,
    )
    shared = auth_database(
        source,
        {
            "anthropic:shared": {
                "type": "api_key",
                "keyRef": {"source": "env", "provider": "default", "id": "OPENAI_API_KEY"},
            },
            "login": {"type": "oauth", "access": "oauth-not-portable"},
        },
        shared=True,
    )
    legacy = source / "agents/main/agent/auth-profiles.json"
    legacy.write_text(
        json.dumps({"profiles": {"retired": {"type": "api_key", "key": "retired-secret"}}})
    )
    try:
        plan = inspect_openclaw(
            source, "current", credentials={"agent:auth:openai:work": "OPENAI_API_KEY"}
        )
        catalog = {item.source: item.importable for item in plan.available_credentials}
        assert catalog["agent:auth:openai:work"]
        assert catalog["shared:auth:anthropic:shared"]
        assert catalog["shared:auth:login"] is False
        assert all("retired" not in item and "private" not in item for item in catalog)
        assert any(file.path.endswith("-wal") for file in plan.files)
        await apply_openclaw(plan)
        assert (
            dotenv_values(profile_root("current") / "credentials.env")["OPENAI_API_KEY"]
            == "wal-secret"
        )
        assert "wal-secret" not in plan.model_dump_json()
    finally:
        local.close()
        shared.close()


def test_legacy_format_supported_only_without_current_store(source):
    legacy = source / "agents/main/agent/auth-profiles.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {"openai:default": {"type": "api_key", "key": "legacy-secret"}},
            }
        )
    )
    plan = inspect_openclaw(
        source, "legacy", credentials={"legacy:auth:openai:default": "OPENAI_API_KEY"}
    )
    assert len(plan.credentials) == 1
    with pytest.raises(MigrationError, match="nonportable"):
        inspect_openclaw(source, "missing", credentials={"dotenv:ABSENT": "OPENAI_API_KEY"})


@pytest.mark.asyncio
async def test_stale_or_tampered_plan_and_existing_profile_cannot_overwrite(source):
    plan = inspect_openclaw(source, "reviewed")
    (source / "workspace/MEMORY.md").write_text("Changed after review")
    with pytest.raises(MigrationError, match="stale"):
        await apply_openclaw(plan)
    assert not profile_root("reviewed").exists()
    plan = inspect_openclaw(source, "reviewed")
    forged = plan.model_copy(deep=True)
    forged.settings["default"]["model"] = "unreviewed-model"
    with pytest.raises(MigrationError, match="modified"):
        await apply_openclaw(forged)
    await apply_openclaw(plan)
    marker = profile_root("reviewed") / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(MigrationError, match="already exists"):
        await apply_openclaw(plan)
    assert marker.read_text() == "unchanged"


@pytest.mark.asyncio
async def test_failed_memory_write_rolls_back_unpublished_profile(source, monkeypatch):
    async def fail(*args, **kwargs):
        raise OSError("test disk full")

    monkeypatch.setattr(SQLiteStorage, "save_scoped_memory", fail)
    with pytest.raises(OSError):
        await apply_openclaw(inspect_openclaw(source, "broken"))
    assert not profile_root("broken").exists()
    assert not list(profile_root("broken").parent.glob(".openclaw-import-*"))


def test_openclaw_gating_and_disabled_skills_archive_instead_of_silently_enabling(source):
    skill(source, "needs-binary", "metadata: {openclaw: {requires: {bins: [magic]}}}\n")
    skill(source, "direct-dispatch", "command-dispatch: tool\ncommand-tool: send_message\n")
    skill(source, "manual-only", "disable-model-invocation: true\n")
    skill(source / "workspace", "portable-guide")
    plan = inspect_openclaw(source, "guides")
    assert plan.installed_skills == ["portable-guide"]
    assert plan.archived_skills == ["direct-dispatch", "manual-only", "needs-binary"]


@pytest.mark.parametrize("kind", ["symlink", "fifo", "nested-symlink", "directory"])
def test_source_file_boundaries_fail_without_hanging_or_reading_target(source, tmp_path, kind):
    target = source / "workspace/MEMORY.md"
    target.unlink()
    if kind == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_text("outside-private-content")
        target.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("outside-private-content")
        (source / "workspace/memory/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MigrationError):
        inspect_openclaw(source, "unsafe")


def test_configured_workspace_traversal_and_symlink_require_explicit_selection(source, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "MEMORY.md").write_text("Explicit external memory")
    config = source / "openclaw.json"
    config.write_text(json.dumps({"agents": {"defaults": {"workspace": str(external)}}}))
    with pytest.raises(MigrationError, match="explicitly"):
        inspect_openclaw(source, "outside")
    assert inspect_openclaw(source, "outside", workspace=external).memory_files == ["MEMORY.md"]
    (source / "linked").symlink_to(external, target_is_directory=True)
    config.write_text(json.dumps({"agents": {"defaults": {"workspace": "linked/nested"}}}))
    (external / "nested").mkdir()
    with pytest.raises(MigrationError):
        inspect_openclaw(source, "linked")
    config.write_text(json.dumps({"agents": {"defaults": {"workspace": "../external"}}}))
    with pytest.raises(MigrationError, match="explicitly"):
        inspect_openclaw(source, "escape")


@pytest.mark.parametrize("variable", ["HOME", "PATH", "HARNESS_HOME", "BAD-NAME"])
def test_reserved_credential_variables_fail_closed(source, variable):
    with pytest.raises(MigrationError):
        inspect_openclaw(source, "vars", credentials={"dotenv:OPENAI_API_KEY": variable})


def test_cli_is_review_then_apply_and_does_not_print_parser_secrets(source, tmp_path):
    app = typer.Typer()
    app.add_typer(migration_app, name="migrate")
    runner = CliRunner()
    output = tmp_path / "plan.json"
    args = [
        "migrate",
        "openclaw",
        "inspect",
        "--source",
        str(source),
        "--profile",
        "cli",
        "--output",
        str(output),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert not profile_root("cli").exists()
    assert "selected-secret" not in result.output
    assert runner.invoke(app, args).exit_code == 1
    result = runner.invoke(app, ["migrate", "openclaw", "apply", "--plan", str(output)])
    assert result.exit_code == 0, result.output
    assert profile_root("cli").exists()
    (source / "openclaw.json").write_text('{apiKey: "very-private-parser-value", invalid ???}')
    result = runner.invoke(app, [*args[:-1], str(tmp_path / "bad.json")])
    assert result.exit_code == 1 and "very-private-parser-value" not in result.output
