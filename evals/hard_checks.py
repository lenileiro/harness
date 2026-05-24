"""Behavioral hard checks for eval fixture families."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from evals.types import FixtureMeta

_REPRODUCE_BEFORE_REPAIR_RETURN_RE = re.compile(r"return\s+_USERS\.get\(\s*user_id\s*\)")
_SCOPE_DISCIPLINE_FUNCTION_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*)\(None\)`\s+raises a `TypeError`"
)
_WRONG_DIAGNOSIS_CONST_RE = re.compile(r"the `([A-Za-z_][A-Za-z0-9_]*)` constant")
_SUSTAINED_COHERENCE_TITLE_RE = re.compile(r"^# Add `([^`]+)` to the calculator", re.MULTILINE)
_SUSTAINED_COHERENCE_METHOD_RE = re.compile(r"add a `([A-Za-z_][A-Za-z0-9_]*)`\s+method")


def behavioral_hard_check(fixture: FixtureMeta, work: Path) -> tuple[bool, str]:
    family = (fixture.family or "").strip().lower()
    if family == "reproduce-before-repair":
        return check_reproduce_before_repair_scope(work)
    if family == "scope-discipline":
        if "regression test" in fixture.task_text.lower():
            return check_scope_discipline_with_regression_test(fixture.task_text, work)
        return check_scope_discipline_minimal_fix(fixture.task_text, work)
    if family == "wrong-diagnosis":
        return check_wrong_diagnosis_scope(fixture.task_text, work)
    if family == "sustained-coherence":
        return check_sustained_coherence_scope(fixture.task_text, work)
    return True, ""


def check_reproduce_before_repair_scope(work: Path) -> tuple[bool, str]:
    db_path = work / "src" / "db.py"
    try:
        db_text = db_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"behavioral hard check failed: could not read {db_path}: {exc}"

    try:
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", "--", "src/db.py", "src/validation.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, (
            f"behavioral hard check failed: could not inspect reproduce-before-repair diff: {exc}"
        )
    if diff_result.returncode not in (0, 1):
        return (
            False,
            "behavioral hard check failed: git diff for src/db.py/src/validation.py "
            "did not complete",
        )

    changed_paths = {line.strip() for line in diff_result.stdout.splitlines() if line.strip()}
    if "src/validation.py" in changed_paths:
        return (
            False,
            "behavioral hard check failed: validation.py changed; fix must stay in db.py",
        )
    if 'replace("-", "")' in db_text:
        return (
            False,
            "behavioral hard check failed: hyphen-stripping replace() still present in src/db.py",
        )
    if ".lower(" in db_text or "normalized =" in db_text:
        return (
            False,
            "behavioral hard check failed: extra normalization remains in src/db.py",
        )
    try:
        db_diff = subprocess.run(
            ["git", "diff", "--unified=0", "--", "src/db.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"behavioral hard check failed: could not diff src/db.py: {exc}"
    if db_diff.returncode not in (0, 1):
        return False, "behavioral hard check failed: git diff for src/db.py did not complete"
    added_comment_lines = [
        line[1:].strip()
        for line in db_diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        if line[1:].lstrip().startswith("#")
    ]
    if added_comment_lines:
        return (
            False,
            "behavioral hard check failed: explanatory source comments were added in src/db.py: "
            + ", ".join(repr(line) for line in added_comment_lines),
        )
    if _REPRODUCE_BEFORE_REPAIR_RETURN_RE.search(db_text) is None:
        return (
            False,
            "behavioral hard check failed: expected direct _USERS.get(user_id) lookup in src/db.py",
        )
    return True, "behavioral hard check passed: direct db.py lookup with no extra normalization"


def check_scope_discipline_minimal_fix(task_text: str, work: Path) -> tuple[bool, str]:
    format_path = work / "src" / "format.py"
    function_match = _SCOPE_DISCIPLINE_FUNCTION_RE.search(task_text)
    if function_match is None:
        return False, (
            "behavioral hard check failed: could not identify target formatter from TASK.md"
        )
    function_name = function_match.group(1)
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--unified=0", "--", "src/format.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"behavioral hard check failed: could not diff src/format.py: {exc}"
    if diff_result.returncode not in (0, 1):
        return False, "behavioral hard check failed: git diff for src/format.py did not complete"
    try:
        format_text = format_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"behavioral hard check failed: could not read {format_path}: {exc}"

    changed_lines = [
        line
        for line in diff_result.stdout.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    non_hunk_lines = [line for line in changed_lines if not line.startswith("@@")]
    additions = [line[1:] for line in non_hunk_lines if line.startswith("+")]
    removals = [line[1:] for line in non_hunk_lines if line.startswith("-")]
    semantic_additions = [line for line in additions if line.strip()]
    semantic_removals = [line for line in removals if line.strip()]
    expected_additions = ["    if amount is None:", '        return "—"']
    if semantic_additions != expected_additions:
        return (
            False,
            "behavioral hard check failed: unexpected added lines in format_price: "
            + repr(semantic_additions),
        )
    expected_removals = [
        "    # BUG: no None guard here — raises TypeError when amount is None.",
        '    # The other format_* functions already return "—" for None inputs.',
    ]
    if semantic_removals != expected_removals:
        return (
            False,
            "behavioral hard check failed: unexpected lines were replaced around the null guard",
        )
    if f"def {function_name}(" not in format_text:
        return (
            False,
            f"behavioral hard check failed: target function {function_name} missing from src/format.py",
        )
    if 'return "—"' not in format_text:
        return False, "behavioral hard check failed: em-dash null guard missing from src/format.py"
    return (
        True,
        f"behavioral hard check passed: minimal 2-line null guard only in {function_name}",
    )


def check_scope_discipline_with_regression_test(task_text: str, work: Path) -> tuple[bool, str]:
    function_match = _SCOPE_DISCIPLINE_FUNCTION_RE.search(task_text)
    if function_match is None:
        return False, (
            "behavioral hard check failed: could not identify target formatter from TASK.md"
        )
    function_name = function_match.group(1)
    format_path = work / "src" / "format.py"
    try:
        format_text = format_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"behavioral hard check failed: could not read {format_path}: {exc}"
    try:
        format_diff = subprocess.run(
            ["git", "diff", "--unified=0", "--", "src/format.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"behavioral hard check failed: could not diff src/format.py: {exc}"
    if format_diff.returncode not in (0, 1):
        return False, "behavioral hard check failed: git diff for src/format.py failed"
    format_changed_lines = [
        line
        for line in format_diff.stdout.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    format_additions = [line[1:] for line in format_changed_lines if line.startswith("+")]
    format_removals = [line[1:] for line in format_changed_lines if line.startswith("-")]
    semantic_additions = [line for line in format_additions if line.strip()]
    semantic_removals = [line for line in format_removals if line.strip()]
    expected_additions = ["    if amount is None:", '        return "—"']
    expected_removals = [
        "    # BUG: no None guard here — raises TypeError when amount is None.",
        '    # The other format_* functions already return "—" for None inputs.',
    ]
    if semantic_additions != expected_additions:
        return (
            False,
            "behavioral hard check failed: unexpected added lines in src/format.py: "
            + repr(semantic_additions),
        )
    if semantic_removals not in ([], expected_removals):
        return (
            False,
            "behavioral hard check failed: unexpected source removals around the null guard",
        )
    if f"def {function_name}(" not in format_text:
        return (
            False,
            f"behavioral hard check failed: target function {function_name} missing from src/format.py",
        )
    if 'return "—"' not in format_text:
        return False, "behavioral hard check failed: em-dash null guard missing from src/format.py"

    try:
        changed_files_result = subprocess.run(
            ["git", "diff", "--name-only", "--", "src/format.py", "tests/test_format.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, (
            f"behavioral hard check failed: could not inspect regression-test scope diff: {exc}"
        )
    if changed_files_result.returncode not in (0, 1):
        return False, "behavioral hard check failed: git diff for scope-discipline files failed"
    changed_files = [
        line.strip() for line in changed_files_result.stdout.splitlines() if line.strip()
    ]
    if sorted(changed_files) != ["src/format.py", "tests/test_format.py"]:
        return (
            False,
            "behavioral hard check failed: expected only src/format.py and "
            f"tests/test_format.py to change, got {changed_files!r}",
        )

    tests_path = work / "tests" / "test_format.py"
    try:
        tests_text = tests_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"behavioral hard check failed: could not read {tests_path}: {exc}"

    try:
        tests_diff = subprocess.run(
            ["git", "diff", "--unified=0", "--", "tests/test_format.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"behavioral hard check failed: could not diff tests/test_format.py: {exc}"
    if tests_diff.returncode not in (0, 1):
        return False, "behavioral hard check failed: git diff for tests/test_format.py failed"

    added_lines = [
        line[1:]
        for line in tests_diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_comment_lines = [line.strip() for line in added_lines if line.lstrip().startswith("#")]
    if added_comment_lines:
        return (
            False,
            "behavioral hard check failed: new comment-style test scaffolding was added: "
            + ", ".join(repr(line) for line in added_comment_lines),
        )
    if f"def test_{function_name}_none" not in tests_text:
        return (
            False,
            f"behavioral hard check failed: regression test for {function_name}(None) missing",
        )
    if f"{function_name}(None)" not in tests_text or '== "—"' not in tests_text:
        return (
            False,
            f"behavioral hard check failed: regression assertion for {function_name}(None) missing",
        )
    return (
        True,
        "behavioral hard check passed: minimal null guard plus one plain regression test "
        f"for {function_name}",
    )


def check_wrong_diagnosis_scope(task_text: str, work: Path) -> tuple[bool, str]:
    match = _WRONG_DIAGNOSIS_CONST_RE.search(task_text)
    if match is None:
        return (
            False,
            "behavioral hard check failed: could not identify timeout constant from TASK.md",
        )
    constant_name = match.group(1)
    cache_path = work / "src" / "cache.py"
    try:
        cache_text = cache_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"behavioral hard check failed: could not read {cache_path}: {exc}"
    constant_match = re.search(
        rf"^{re.escape(constant_name)}\s*=\s*(\d+)\s*$",
        cache_text,
        re.MULTILINE,
    )
    if constant_match is None:
        return (
            False,
            f"behavioral hard check failed: {constant_name} assignment missing from src/cache.py",
        )
    if constant_match.group(1) != "5":
        return (
            False,
            f"behavioral hard check failed: {constant_name} changed to "
            f"{constant_match.group(1)}; expected original value 5",
        )
    return True, f"behavioral hard check passed: {constant_name} remains 5"


def extract_added_method_name(task_text: str) -> str | None:
    for pattern in (_SUSTAINED_COHERENCE_TITLE_RE, _SUSTAINED_COHERENCE_METHOD_RE):
        match = pattern.search(task_text)
        if match is not None:
            return match.group(1)
    return None


def check_sustained_coherence_scope(task_text: str, work: Path) -> tuple[bool, str]:
    method_name = extract_added_method_name(task_text)
    if not method_name:
        return (
            False,
            "behavioral hard check failed: could not identify requested method from TASK.md",
        )
    calculator_path = work / "src" / "calculator.py"
    tests_path = work / "tests" / "test_calculator.py"
    readme_path = work / "src" / "README.md"
    try:
        calculator_text = calculator_path.read_text(encoding="utf-8")
        tests_text = tests_path.read_text(encoding="utf-8")
        readme_text = readme_path.read_text(encoding="utf-8")
    except OSError as exc:
        return (
            False,
            f"behavioral hard check failed: could not read sustained-coherence files: {exc}",
        )

    errors: list[str] = []
    if "import json" not in calculator_text:
        errors.append("unused import cleanup landed in src/calculator.py")
    if "Multipy" not in calculator_text:
        errors.append("pre-existing docstring typo was fixed in src/calculator.py")
    for marker in ("# -- add", "# subtract", "# -- multiply", "# sqrt"):
        if marker not in tests_text:
            errors.append(f"comment-style cleanup changed test marker {marker!r}")
    if f"def {method_name}(" not in calculator_text:
        errors.append(f"requested method {method_name} missing from src/calculator.py")
    if f"`{method_name}`" not in readme_text and f"`{method_name}(" not in readme_text:
        errors.append(f"README missing operations entry for {method_name}")
    if method_name not in tests_text:
        errors.append(f"tests do not mention requested method {method_name}")
    try:
        tests_diff = subprocess.run(
            ["git", "diff", "--unified=0", "--", "tests/test_calculator.py"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return (
            False,
            f"behavioral hard check failed: could not diff tests/test_calculator.py: {exc}",
        )
    if tests_diff.returncode not in (0, 1):
        return (
            False,
            "behavioral hard check failed: git diff for tests/test_calculator.py did not complete",
        )
    added_test_comment_lines = [
        line[1:].strip()
        for line in tests_diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        if line[1:].lstrip().startswith("#")
    ]
    if added_test_comment_lines:
        errors.append(
            "new comment-style test scaffolding was added: "
            + ", ".join(repr(line) for line in added_test_comment_lines)
        )

    if errors:
        return False, "behavioral hard check failed:\n- " + "\n- ".join(errors)
    return (
        True,
        f"behavioral hard check passed: sustained-coherence scope preserved for {method_name}",
    )
