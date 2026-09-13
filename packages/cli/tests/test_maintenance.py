import io
import sqlite3
import tarfile

import pytest

from harness.cli.maintenance_commands import create_backup, restore_backup


def test_backup_consistent_sqlite_and_private_credential_policy(tmp_path):
    source = tmp_path / "identity"
    source.mkdir()
    (source / "config.toml").write_text("[default]\nmodel='example'\n")
    (source / "credentials.env").write_text("TOKEN='private'\n")
    (source / "auth").mkdir()
    (source / "auth/tokens.json").write_text('{"token":"private"}')
    for name in [
        "aws/credentials",
        "gcloud/application_default_credentials.json",
        "azure/msal_token_cache.json",
        "huggingface/token",
        "modal.toml",
        "workspace/.harness/channels/weixin-account.json",
        "workspace/.harness/channels/photon-account.json",
    ]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private-sdk-credential")
    db = sqlite3.connect(source / "sessions.db")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE data(value TEXT)")
    db.execute("INSERT INTO data VALUES ('committed')")
    db.commit()
    try:
        archive = tmp_path / "snapshot.tar.gz"
        manifest = create_backup(source, archive)
        assert not manifest["credentials_included"]
        assert {item["path"] for item in manifest["files"]} == {"sessions.db", "config.toml"}
        target = tmp_path / "restored"
        restore_backup(archive, target)
        restored = sqlite3.connect(target / "sessions.db")
        try:
            assert restored.execute("SELECT value FROM data").fetchone() == ("committed",)
        finally:
            restored.close()
        assert not (target / "credentials.env").exists()
        with pytest.raises(ValueError, match="must not exist"):
            restore_backup(archive, target)
    finally:
        db.close()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("linked", tarfile.SYMTYPE),
        ("C:/escape", tarfile.REGTYPE),
    ],
)
def test_restore_rejects_unsafe_entries_before_creating_destination(tmp_path, name, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(name)
        info.type = kind
        info.size = 1 if kind == tarfile.REGTYPE else 0
        info.linkname = "../escape" if kind == tarfile.SYMTYPE else ""
        bundle.addfile(info, io.BytesIO(b"x") if info.size else None)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="unsafe"):
        restore_backup(archive, destination)
    assert not destination.exists()


def test_cli_backup_captures_xdg_database_config_and_external_workspace(tmp_path, monkeypatch):
    import typer
    from typer.testing import CliRunner

    import harness.storage.sqlite as sqlite
    from harness.cli import config
    from harness.cli import maintenance_commands as commands

    home = tmp_path / "identity"
    home.mkdir()
    workspace = tmp_path / "work"
    private = workspace / ".harness"
    private.mkdir(parents=True)
    (private / "notes.md").write_text("workspace state")
    (private / "credentials.env").write_text("TOKEN=private")
    external = tmp_path / "xdg"
    external.mkdir()
    config_file = external / "config.toml"
    config_file.write_text('[default]\nmodel="example"\n')
    database = external / "sessions.db"
    database.write_bytes(b"fixture database")
    monkeypatch.setattr(commands, "user_home", lambda: home)
    monkeypatch.setattr(config, "default_config_path", lambda: config_file)
    monkeypatch.setattr(sqlite, "default_db_path", lambda: database)
    monkeypatch.chdir(workspace)
    app = typer.Typer()
    app.add_typer(commands.maintenance_app, name="maintenance")
    archive = tmp_path / "backup.tar.gz"
    result = CliRunner().invoke(app, ["maintenance", "backup", str(archive)])
    assert result.exit_code == 0, result.output
    restored = tmp_path / "restored"
    restore_backup(archive, restored)
    assert (restored / "config.toml").read_text() == config_file.read_text()
    assert (restored / "state/sessions.db").read_bytes() == b"fixture database"
    assert (restored / "workspace/.harness/notes.md").read_text() == "workspace state"
    assert not (restored / "workspace/.harness/credentials.env").exists()
