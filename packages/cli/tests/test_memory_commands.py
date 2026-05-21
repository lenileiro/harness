"""Tests for `harness memory` CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main


def _run(args: list[str]):
    return CliRunner().invoke(cli_main.app, args)


class TestMemorySave:
    def test_save_default_kind(self) -> None:
        r = _run(["memory", "save", "uses uv workspace", "--in-memory"])
        assert r.exit_code == 0, r.stdout
        assert "Saved" in r.stdout
        assert "mem_" in r.stdout

    def test_save_explicit_kind(self) -> None:
        r = _run(["memory", "save", "prefers concise", "--kind", "user_preference", "--in-memory"])
        assert r.exit_code == 0, r.stdout
        assert "user_preference" in r.stdout

    def test_save_all_kinds(self) -> None:
        for kind in ("user_preference", "user_fact", "project_fact", "project_context"):
            r = _run(["memory", "save", f"test {kind}", "--kind", kind, "--in-memory"])
            assert r.exit_code == 0, r.stdout

    def test_save_invalid_kind(self) -> None:
        r = _run(["memory", "save", "something", "--kind", "bogus", "--in-memory"])
        assert r.exit_code == 1
        assert "Invalid" in r.stdout


class TestMemoryList:
    def test_list_empty(self) -> None:
        r = _run(["memory", "list", "--in-memory"])
        assert r.exit_code == 0, r.stdout
        assert "No memories" in r.stdout

    def test_list_shows_entries(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        _run(["memory", "save", "uses uv", "--kind", "project_fact", "--db", db])
        _run(["memory", "save", "concise please", "--kind", "user_preference", "--db", db])

        r = _run(["memory", "list", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "uses uv" in r.stdout
        assert "concise please" in r.stdout

    def test_list_filter_by_kind(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        _run(["memory", "save", "uses uv", "--kind", "project_fact", "--db", db])
        _run(["memory", "save", "concise please", "--kind", "user_preference", "--db", db])

        r = _run(["memory", "list", "--kind", "project_fact", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "uses uv" in r.stdout
        assert "concise please" not in r.stdout

    def test_list_invalid_kind(self) -> None:
        r = _run(["memory", "list", "--kind", "bogus", "--in-memory"])
        assert r.exit_code == 1


class TestMemorySearch:
    def test_search_finds_match(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        _run(["memory", "save", "uses uv workspace", "--kind", "project_fact", "--db", db])
        _run(["memory", "save", "python version 3.12", "--kind", "project_fact", "--db", db])

        r = _run(["memory", "search", "uv", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "uses uv workspace" in r.stdout
        assert "python version" not in r.stdout

    def test_search_no_results(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        _run(["memory", "save", "something else", "--kind", "project_fact", "--db", db])

        r = _run(["memory", "search", "xyznotfound", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "No matches" in r.stdout


class TestMemoryRm:
    def test_rm_deletes_entry(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        save_r = _run(["memory", "save", "to delete", "--kind", "project_fact", "--db", db])
        assert save_r.exit_code == 0
        entry_id = next(w for w in save_r.stdout.split() if w.startswith("mem_"))

        r = _run(["memory", "rm", entry_id, "--db", db, "--yes"])
        assert r.exit_code == 0, r.stdout
        assert "Deleted" in r.stdout

        list_r = _run(["memory", "list", "--db", db])
        assert "to delete" not in list_r.stdout

    def test_rm_not_found(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        r = _run(["memory", "rm", "mem_doesnotexist", "--db", db, "--yes"])
        assert r.exit_code == 1
        assert "not found" in r.stdout.lower()
