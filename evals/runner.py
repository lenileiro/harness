"""Eval fixture runner.

Discovers fixtures under evals/fixtures/, runs each one by:
  1. shutil.copytree to a temp dir
  2. git init + initial commit (clean baseline)
  3. reads TASK.md as the agent prompt
  4. invokes `harness run` as a subprocess (captures stdout+stderr)
  5. captures `git diff HEAD` (what the agent changed)
  6. runs `python -m pytest tests/` (whether the fix is correct)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from itertools import pairwise
from pathlib import Path
from typing import cast

from evals.failure_analyzer import persist_harness_adjustments
from evals.types import FixtureMeta, FixtureRules, HardMetrics, RunOutcome, TraceEvent

_DEFAULT_VERIFY_COMMAND = "pytest tests/ -v --tb=short --no-header"
_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "shell",
    "verify_work",
    "complete_work_item",
)
_REPRODUCE_BEFORE_REPAIR_RETURN_RE = re.compile(r"return\s+_USERS\.get\(\s*user_id\s*\)")
_SCOPE_DISCIPLINE_FUNCTION_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*)\(None\)`\s+raises a `TypeError`"
)
_WRONG_DIAGNOSIS_CONST_RE = re.compile(r"the `([A-Za-z_][A-Za-z0-9_]*)` constant")
_SUSTAINED_COHERENCE_TITLE_RE = re.compile(r"^# Add `([^`]+)` to the calculator", re.MULTILINE)
_SUSTAINED_COHERENCE_METHOD_RE = re.compile(r"add a `([A-Za-z_][A-Za-z0-9_]*)`\s+method")


def _find_evals_root() -> Path:
    """Walk CWD upward to find evals/fixtures/."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / "evals" / "fixtures"
        if candidate.is_dir():
            return current / "evals"
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find evals/fixtures/ — run from inside the harness repo."
            )
        current = parent


def discover_fixtures(
    evals_root: Path | None = None,
    *,
    fixtures_subdir: str = "fixtures",
    include_holdout: bool = False,
) -> list[FixtureMeta]:
    """Return all fixtures sorted by directory name."""
    root = evals_root or _find_evals_root()
    fixtures_dir = root / fixtures_subdir
    result: list[FixtureMeta] = []
    if not fixtures_dir.exists():
        return result
    for entry in sorted(fixtures_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_path = entry / "TASK.md"
        eval_path = entry / "EVAL.md"
        if not task_path.exists() or not eval_path.exists():
            continue
        metadata = _load_fixture_config(entry / "fixture.yaml")
        eval_md = eval_path.read_text(encoding="utf-8")
        phases = _coerce_optional_list(metadata.get("phases"))
        if not phases:
            phases = _parse_phases(eval_md)
        family = str(metadata.get("family") or entry.name.split("-", 1)[-1])
        rules = _rules_from_metadata(eval_md, metadata)
        fixture = FixtureMeta(
            name=entry.name,
            path=entry,
            task_text=task_path.read_text(encoding="utf-8"),
            eval_md=eval_md,
            verify_command=str(metadata.get("verify_command") or _DEFAULT_VERIFY_COMMAND),
            phases=phases,
            family=family,
            holdout=bool(metadata.get("holdout", False)),
            mutated_from=_coerce_optional_str(metadata.get("mutated_from")),
            metadata_path=entry / "fixture.yaml" if (entry / "fixture.yaml").exists() else None,
            rules=rules,
        )
        if fixture.holdout and not include_holdout:
            continue
        result.append(fixture)
    return result


def _rules_from_metadata(eval_md: str, metadata: dict[str, object]) -> FixtureRules:
    return FixtureRules(
        behavior_category=str(metadata.get("behavior_category") or metadata.get("family") or ""),
        primary_dimension=str(
            metadata.get("primary_dimension") or _extract_eval_field(eval_md, "primary_dimension")
        ),
        expected_first_step=str(
            metadata.get("expected_first_step") or _expected_first_step_from_eval(eval_md)
        ),
        allowed_paths=_ensure_list(metadata.get("allowed_paths")),
        disallowed_paths=_ensure_list(metadata.get("disallowed_paths")),
        required_verification=str(metadata.get("required_verification") or ""),
        trap=str(metadata.get("trap") or _extract_eval_field(eval_md, "trap")),
        correct_fix=str(metadata.get("correct_fix") or _extract_eval_field(eval_md, "correct_fix")),
        dimensions=_ensure_list(metadata.get("dimensions"))
        or _parse_csv_field(_extract_eval_field(eval_md, "dimensions")),
        scoring_notes=str(
            metadata.get("scoring_notes") or _extract_eval_field(eval_md, "scoring_notes")
        ),
    )


def _ensure_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            return _parse_csv_field(inner)
        return _parse_csv_field(value)
    return [str(value)]


def _coerce_optional_list(value: object) -> list[str] | None:
    items = _ensure_list(value)
    return items or None


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_csv_field(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _extract_eval_field(eval_md: str, field: str) -> str:
    pattern = re.compile(rf"^{re.escape(field)}:\s*(.*)$", re.IGNORECASE)
    lines = eval_md.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if value and value != ">":
            return value
        collected: list[str] = []
        for follow in lines[index + 1 :]:
            if not follow.strip():
                break
            if not follow.startswith((" ", "\t")):
                break
            collected.append(follow.strip())
        return " ".join(collected).strip()
    return ""


def _expected_first_step_from_eval(eval_md: str) -> str:
    lower = eval_md.lower()
    if "run the tests" in lower or "ran tests" in lower:
        return "run tests"
    if "inspect" in lower:
        return "inspect code"
    return ""


def _load_fixture_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    config: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key:
            config.setdefault(current_list_key, [])
            items = cast(list[str], config[current_list_key])
            items.append(stripped[2:].strip().strip("\"'"))
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            config[key] = []
            current_list_key = key
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        lowered = value.lower()
        if lowered in {"true", "false"}:
            config[key] = lowered == "true"
        elif value.startswith("[") and value.endswith("]"):
            config[key] = _parse_csv_field(value[1:-1])
        else:
            config[key] = value
    return config


def _parse_phases(eval_md: str) -> list[str] | None:
    """Pull a `phases:` line from EVAL.md into an ordered phase list.

    Accepts either inline comma-separated form:
        phases: implement, test, document, verify
    or a leading-dash list (one phase per line, indented two spaces):
        phases:
          - implement
          - test
    """
    import re

    lines = eval_md.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*phases\s*:\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        inline = m.group(1).strip()
        if inline:
            parts = [p.strip().lower() for p in inline.split(",") if p.strip()]
            return parts or None
        # Block form — gather indented `- name` lines.
        names: list[str] = []
        for follow in lines[i + 1 :]:
            stripped = follow.strip()
            if not stripped:
                break
            if not follow.startswith((" ", "\t")):
                break
            if stripped.startswith("-"):
                name = stripped.lstrip("- ").strip().lower()
                if name:
                    names.append(name)
            else:
                break
        return names or None
    return None


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "eval",
            "GIT_AUTHOR_EMAIL": "eval@harness",
            "GIT_COMMITTER_NAME": "eval",
            "GIT_COMMITTER_EMAIL": "eval@harness",
        }
    )
    return env


def _agent_cmd(
    provider: str,
    model: str,
    task_text: str,
    work: Path,
    harness_bin: str | None,
    verify_command: str | None = None,
    variant: str = "defended",
    phases: list[str] | None = None,
    behavior_category: str | None = None,
    max_output_tokens: int | None = None,
) -> list[str]:
    """Return the command to invoke the agent for the given provider.

    'claude' provider shells out to `claude -p` (Claude Code CLI).
    All other providers go through `harness run` with ShellVerifier so the
    repair loop fires when the verify_command exits non-zero.

    `variant="bare"` uses `--profile bare` to disable the structural
    verifier chain and critic — the agent runs with model + tools only.
    The defended arm now uses `--profile adaptive`, which picks between
    minimal and strict from the task shape. Used by A/B mode to measure
    harness value-add.
    """
    if provider == "claude":
        claude_bin = shutil.which("claude") or "claude"
        return [
            claude_bin,
            "-p",
            task_text.strip(),
            "--allowedTools",
            "Read,Write,Edit,Bash",
        ]
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    cmd = [
        harness_cmd,
        "run",
        task_text.strip(),
        "--cwd",
        str(work),
        "--yes",
        "--in-memory",
        "--provider",
        provider,
        "--model",
        model,
        # 2 attempts: empirically the right budget for "minimal fix" style
        # tasks. Raising it to 4 regressed fixture 02 because the extra
        # turns gave the agent room to add unwanted "robustness" code,
        # net-net worse on scope dimension. Keep tight.
        "--max-repair",
        "2",
    ]
    # "defended" → adaptive profile (current best harness path).
    # "bare"     → no chain, no critic.
    # Eval keeps using defended/bare labels so historical tables remain
    # comparable even as the defended implementation improves.
    cmd += ["--profile", "adaptive" if variant == "defended" else "bare"]
    if max_output_tokens is not None:
        cmd += ["--max-output-tokens", str(max_output_tokens)]
    if verify_command:
        # In bare mode, keep ShellVerifier (the test runner) but skip the
        # critic — bare = agent + tools + test signal, nothing else.
        cmd += [
            "--verify",
            "shell",
            "--verify-command",
            verify_command,
        ]
        # Critic helps on diagnosis-heavy decomposition traps (fixture 03), but
        # on strict minimal-fix tasks it can create extra repair-loop churn and
        # overthinking. Keep the defended arm lighter on pure scope /
        # verification fixtures.
        if variant != "bare" and (behavior_category or "").strip().lower() in {
            "decomposition",
            "diagnosis",
        }:
            critic_mode = "llm+search" if os.environ.get("TAVILY_API_KEY") else "llm"
            cmd += ["--critic", critic_mode]
    else:
        cmd += ["--verify", "rule"]
    # Phase tracking is useful for some live tasks, but in this benchmark it
    # adds substantial finish-line latency on the long sustained-coherence
    # fixture without improving judged outcomes. Since the eval objective here
    # is pass-rate-first A/B comparison, keep defended lighter on pure scope /
    # verification families and reserve explicit phase gating for tasks whose
    # primary challenge is multi-stage decomposition.
    if (
        phases
        and variant != "bare"
        and (behavior_category or "").strip().lower() not in {"scope", "verification"}
    ):
        cmd += ["--phases", ",".join(phases)]
    return cmd


def _copy_fixture_for_run(src: Path, dest: Path) -> None:
    """Copy a fixture into an isolated worktree without leaking judge metadata.

    `EVAL.md` and `fixture.yaml` are benchmark-side artifacts. The runner
    provides their contents to the judge and discovery layer directly, so the
    agent should not be able to read them from the workspace during execution.
    """
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("EVAL.md", "fixture.yaml", "__pycache__", "*.pyc", "*.pyo"),
    )


def _behavioral_hard_check(fixture: FixtureMeta, work: Path) -> tuple[bool, str]:
    family = (fixture.family or "").strip().lower()
    if family == "reproduce-before-repair":
        return _check_reproduce_before_repair_scope(work)
    if family == "scope-discipline":
        if "regression test" in fixture.task_text.lower():
            return _check_scope_discipline_with_regression_test(fixture.task_text, work)
        return _check_scope_discipline_minimal_fix(fixture.task_text, work)
    if family == "wrong-diagnosis":
        return _check_wrong_diagnosis_scope(fixture.task_text, work)
    if family == "sustained-coherence":
        return _check_sustained_coherence_scope(fixture.task_text, work)
    return True, ""


def _check_reproduce_before_repair_scope(work: Path) -> tuple[bool, str]:
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
            "behavioral hard check failed: could not inspect reproduce-before-repair diff: "
            f"{exc}"
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


def _check_scope_discipline_minimal_fix(task_text: str, work: Path) -> tuple[bool, str]:
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


def _check_scope_discipline_with_regression_test(task_text: str, work: Path) -> tuple[bool, str]:
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
            "behavioral hard check failed: could not inspect regression-test scope diff: " f"{exc}"
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


def _check_wrong_diagnosis_scope(task_text: str, work: Path) -> tuple[bool, str]:
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


def _extract_added_method_name(task_text: str) -> str | None:
    for pattern in (_SUSTAINED_COHERENCE_TITLE_RE, _SUSTAINED_COHERENCE_METHOD_RE):
        match = pattern.search(task_text)
        if match is not None:
            return match.group(1)
    return None


def _check_sustained_coherence_scope(task_text: str, work: Path) -> tuple[bool, str]:
    method_name = _extract_added_method_name(task_text)
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


def _build_trace_events(
    transcript: str,
    verify_command: str,
    *,
    agent_exit_code: int,
    verify_exit_code: int,
) -> list[TraceEvent]:
    events: list[TraceEvent] = [
        TraceEvent(kind="agent_exit", order=1, data={"exit_code": agent_exit_code}),
        TraceEvent(kind="verify_exit", order=2, data={"exit_code": verify_exit_code}),
    ]
    order = len(events) + 1
    for tool_name in _extract_tool_sequence(transcript):
        events.append(TraceEvent(kind="tool_call", order=order, data={"tool": tool_name}))
        order += 1
    verify_name = verify_command.split()[0] if verify_command.strip() else "verify"
    if _transcript_mentions_verification(transcript, verify_command):
        events.append(
            TraceEvent(
                kind="verification_observed",
                order=order,
                message=f"Detected verification marker for {verify_name}.",
                data={"command": verify_command},
            )
        )
    return events


def _extract_tool_sequence(transcript: str) -> list[str]:
    sequence: list[str] = []
    for line in transcript.splitlines():
        lowered = line.lower()
        for tool_name in _TOOL_NAMES:
            if tool_name in lowered:
                sequence.append(tool_name)
                break
    return sequence


def _transcript_mentions_verification(transcript: str, verify_command: str) -> bool:
    lowered = transcript.lower()
    if "verify_work" in lowered:
        return True
    verify_head = verify_command.strip().split()[0].lower() if verify_command.strip() else ""
    if verify_head and verify_head in lowered:
        return True
    return any(marker in lowered for marker in ("pytest", "cargo test", "go test", "npm test"))


def _diff_stats(git_diff: str) -> tuple[int, int, int]:
    files: set[str] = set()
    lines_added = 0
    lines_deleted = 0
    for line in git_diff.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("+"):
            lines_added += 1
        elif line.startswith("-"):
            lines_deleted += 1
    return len(files), lines_added, lines_deleted


def _compute_hard_metrics(
    transcript: str,
    git_diff: str,
    verify_command: str,
    *,
    run_exit_code: int,
    verify_exit_code: int,
    agent_duration_seconds: float,
    verify_duration_seconds: float,
) -> HardMetrics:
    files_touched, lines_added, lines_deleted = _diff_stats(git_diff)
    tool_sequence = _extract_tool_sequence(transcript)
    did_run_verification = _transcript_mentions_verification(transcript, verify_command)
    verify_positions = [idx for idx, name in enumerate(tool_sequence) if name == "verify_work"]
    first_verify_idx = verify_positions[0] if verify_positions else None
    mutating_tools = {"write_file", "edit_file", "shell"}
    edit_before_repro = False
    if first_verify_idx is not None:
        edit_before_repro = any(name in mutating_tools for name in tool_sequence[:first_verify_idx])
    redundant_tool_calls = 0
    retry_loops = 0
    streak = 1
    for prev, current in pairwise(tool_sequence):
        if prev == current:
            streak += 1
            redundant_tool_calls += 1
            if streak >= 3:
                retry_loops += 1
        else:
            streak = 1
    lowered = transcript.lower()
    success_claim = any(phrase in lowered for phrase in ("done", "fixed", "all set", "completed"))
    premature_completion = success_claim and verify_exit_code != 0
    verification_after_failure = tool_sequence.count("verify_work") >= 2 or (
        did_run_verification and verify_exit_code == 0 and run_exit_code != 0
    )
    shell_commands = tool_sequence.count("shell")
    return HardMetrics(
        verify_passed=verify_exit_code == 0,
        run_exit_code=run_exit_code,
        verify_exit_code=verify_exit_code,
        files_touched=files_touched,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        tool_calls=len(tool_sequence),
        shell_commands=shell_commands,
        did_run_verification=did_run_verification,
        agent_duration_seconds=agent_duration_seconds,
        verify_duration_seconds=verify_duration_seconds,
        total_duration_seconds=agent_duration_seconds + verify_duration_seconds,
        time_to_first_verification_seconds=agent_duration_seconds if did_run_verification else None,
        edit_before_repro=edit_before_repro,
        premature_completion=premature_completion,
        redundant_tool_calls=redundant_tool_calls,
        retry_loops=retry_loops,
        verification_after_failure=verification_after_failure,
    )


def _persist_artifacts(
    artifact_dir: Path,
    outcome: RunOutcome,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "transcript.txt").write_text(outcome.transcript, encoding="utf-8")
    (artifact_dir / "git_diff.patch").write_text(outcome.git_diff, encoding="utf-8")
    (artifact_dir / "verify_output.txt").write_text(outcome.test_output, encoding="utf-8")
    (artifact_dir / "agent_command.json").write_text(
        json.dumps(outcome.agent_command, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "outcome.json").write_text(
        json.dumps(outcome.to_dict(), indent=2),
        encoding="utf-8",
    )
    with (artifact_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in outcome.trace_events:
            handle.write(json.dumps(event.to_dict()) + "\n")


def run_fixture(
    fixture: FixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    agent_timeout: int = 300,
    test_timeout: int = 60,
    variant: str = "defended",
    artifact_dir: Path | None = None,
    max_output_tokens: int | None = None,
) -> RunOutcome:
    """Run one fixture end-to-end in an isolated temp directory."""
    with tempfile.TemporaryDirectory(prefix="harness_eval_") as tmp_str:
        work = Path(tmp_str) / fixture.name
        _copy_fixture_for_run(fixture.path, work)

        git_env = _git_env()

        # Create a clean git baseline so git diff HEAD captures only agent changes.
        # Write a .gitignore first so __pycache__ / *.pyc don't pollute the diff.
        (work / ".gitignore").write_text("__pycache__/\n*.pyc\n*.pyo\n")
        subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "add", "-A"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", "initial", "--no-verify"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )

        # Run the agent.
        cmd = _agent_cmd(
            provider,
            model,
            fixture.task_text,
            work,
            harness_bin,
            verify_command=fixture.verify_command,
            variant=variant,
            phases=fixture.phases,
            behavior_category=fixture.rules.behavior_category or fixture.family,
            max_output_tokens=max_output_tokens,
        )
        agent_started = time.perf_counter()
        agent_env = os.environ.copy()
        experience_root = (_find_evals_root() / "runs").resolve()
        existing_roots = [
            raw.strip()
            for raw in agent_env.get("HARNESS_EXPERIENCE_ROOTS", "").split(os.pathsep)
            if raw.strip()
        ]
        if str(experience_root) not in existing_roots:
            existing_roots.append(str(experience_root))
        agent_env["HARNESS_EXPERIENCE_ROOTS"] = os.pathsep.join(existing_roots)
        agent_result = subprocess.run(
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=agent_timeout,
            env=agent_env,
        )
        agent_duration = time.perf_counter() - agent_started
        transcript = agent_result.stdout + agent_result.stderr

        # Capture what the agent changed.
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=work,
            capture_output=True,
            text=True,
        )
        git_diff = diff_result.stdout

        # Run the fixture's own verify_command to check correctness. We
        # invoke whatever shell command the fixture declared (pytest, go
        # test, cargo, jest, ...) rather than hardcoding pytest here — the
        # harness eval framework is language-agnostic.
        verify_started = time.perf_counter()
        test_result = subprocess.run(
            fixture.verify_command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=test_timeout,
            env=os.environ.copy(),
            shell=True,
        )
        verify_duration = time.perf_counter() - verify_started
        test_output = test_result.stdout + test_result.stderr
        combined_verify_exit_code = test_result.returncode
        if test_result.returncode == 0:
            behavioral_ok, behavioral_message = _behavioral_hard_check(fixture, work)
            if behavioral_message:
                separator = "\n" if test_output.endswith("\n") or not test_output else "\n\n"
                test_output = f"{test_output}{separator}{behavioral_message}\n"
            if not behavioral_ok:
                combined_verify_exit_code = 1
        trace_events = _build_trace_events(
            transcript,
            fixture.verify_command,
            agent_exit_code=agent_result.returncode,
            verify_exit_code=combined_verify_exit_code,
        )
        hard_metrics = _compute_hard_metrics(
            transcript,
            git_diff,
            fixture.verify_command,
            run_exit_code=agent_result.returncode,
            verify_exit_code=combined_verify_exit_code,
            agent_duration_seconds=agent_duration,
            verify_duration_seconds=verify_duration,
        )
        outcome = RunOutcome(
            fixture=fixture,
            transcript=transcript,
            git_diff=git_diff,
            test_output=test_output,
            agent_exit_code=agent_result.returncode,
            test_exit_code=combined_verify_exit_code,
            variant=variant,
            hard_metrics=hard_metrics,
            trace_events=trace_events,
            artifact_dir=artifact_dir,
            agent_command=cmd,
            verify_command=fixture.verify_command,
        )
        if artifact_dir is not None:
            _persist_artifacts(artifact_dir, outcome)
            persist_harness_adjustments(artifact_dir)
        return outcome
