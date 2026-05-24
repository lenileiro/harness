"""Tests for scope-discipline hard checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner


def test_scope_discipline_behavioral_check_accepts_exact_two_line_fix(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "minimal 2-line null guard only" in message


def test_scope_discipline_behavioral_check_rejects_extra_changes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
            "    rounded = round(value, 2)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
            "    rounded = _round_half_up(value, 2)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is False
    assert "unexpected added lines" in message or "unexpected lines were replaced" in message


def test_scope_discipline_behavioral_check_accepts_extra_blank_line_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
            "    return value\n\n"
            "def format_percentage(value):\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
            "    return value\n\n\n"
            "def format_percentage(value):\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "minimal 2-line null guard only" in message


def test_scope_discipline_behavioral_check_accepts_mutated_function_name(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in render_amount\n\n`render_amount(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "render_amount" in message


def test_scope_discipline_regression_test_check_rejects_comment_scaffolding(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            "# -- render_amount none\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is False
    assert "new comment-style test scaffolding was added" in message


def test_scope_discipline_regression_test_check_accepts_plain_test_addition(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is True
    assert "one plain regression test" in message


def test_scope_discipline_regression_test_check_accepts_preserved_bug_comments(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is True
    assert "one plain regression test" in message
