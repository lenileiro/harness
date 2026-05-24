"""Tests for reproduce-before-repair hard checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner


def test_reproduce_before_repair_behavioral_check_accepts_direct_lookup(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    """Fetch a raw user record by id. Returns None if not found."""\n'
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    """Fetch a raw user record by id. Returns None if not found."""\n'
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is True
    assert "no extra normalization" in message


def test_reproduce_before_repair_behavioral_check_rejects_extra_normalization(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    normalized = user_id.lower()\n"
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "extra normalization remains" in message


def test_reproduce_before_repair_behavioral_check_rejects_explanatory_comment_addition(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    # Normalize the ID before lookup to handle legacy system quirks.\n"
            '    # BUG: this strips hyphens, so "abc-def" becomes "abcdef" — no match.\n'
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    # Normalize the ID before lookup to handle legacy system quirks.\n"
            "    # We no longer strip hyphens as they are a valid part of IDs.\n"
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "explanatory source comments were added" in message


def test_reproduce_before_repair_behavioral_check_rejects_validation_changes(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        "def get_user_record(user_id: str) -> dict | None:\n    return _USERS.get(user_id)\n",
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 2\n", encoding="utf-8")

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "validation.py changed" in message
