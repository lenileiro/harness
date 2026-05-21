"""Tests for workspace-local storage auto-detection and harness init."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli.__main__ import _build_storage, _workspace_db
from harness.storage.sqlite import SQLiteStorage, default_db_path


def _run(args: list[str]):
    return CliRunner().invoke(cli_main.app, args)


class TestWorkspaceDbHelper:
    def test_returns_none_when_no_harness_dir(self, tmp_path: Path) -> None:
        assert _workspace_db(tmp_path) is None

    def test_returns_none_when_dir_exists_but_no_db(self, tmp_path: Path) -> None:
        (tmp_path / ".harness").mkdir()
        assert _workspace_db(tmp_path) is None

    def test_returns_path_when_db_exists(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        db = harness_dir / "harness.db"
        db.touch()
        result = _workspace_db(tmp_path)
        assert result == db


class TestBuildStorage:
    def test_in_memory_ignores_everything(self, tmp_path: Path) -> None:
        from harness.storage.memory import InMemoryStorage

        storage = _build_storage(db=None, in_memory=True, cwd=tmp_path)
        assert isinstance(storage, InMemoryStorage)

    def test_explicit_db_overrides_workspace(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        (harness_dir / "harness.db").touch()

        explicit = tmp_path / "explicit.db"
        storage = _build_storage(db=explicit, in_memory=False, cwd=tmp_path)
        assert isinstance(storage, SQLiteStorage)
        assert storage.path == explicit

    def test_workspace_db_auto_selected(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        workspace_db = harness_dir / "harness.db"
        workspace_db.touch()

        storage = _build_storage(db=None, in_memory=False, cwd=tmp_path)
        assert isinstance(storage, SQLiteStorage)
        assert storage.path == workspace_db

    def test_falls_back_to_xdg_default(self, tmp_path: Path) -> None:
        storage = _build_storage(db=None, in_memory=False, cwd=tmp_path)
        assert isinstance(storage, SQLiteStorage)
        assert storage.path == default_db_path()


class TestHarnessInit:
    def test_creates_harness_dir_and_db(self, tmp_path: Path) -> None:
        r = _run(["init", "--cwd", str(tmp_path)])
        assert r.exit_code == 0, r.stdout
        assert "Initialized" in r.stdout
        assert (tmp_path / ".harness" / "harness.db").exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        _run(["init", "--cwd", str(tmp_path)])
        r = _run(["init", "--cwd", str(tmp_path)])
        assert r.exit_code == 0, r.stdout
        assert "nothing to do" in r.stdout.lower() or "Already" in r.stdout

    def test_workspace_db_auto_used_after_init(self, tmp_path: Path) -> None:
        _run(["init", "--cwd", str(tmp_path)])
        assert (tmp_path / ".harness" / "harness.db").exists()
        storage = _build_storage(db=None, in_memory=False, cwd=tmp_path)
        assert isinstance(storage, SQLiteStorage)
        assert storage.path == tmp_path / ".harness" / "harness.db"
