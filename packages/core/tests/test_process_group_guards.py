"""Repo-wide invariant: every `os.killpg` tolerates an already-vanished group.

Signalling a process group whose leader has already been reaped raises
`ProcessLookupError` (ESRCH) on Linux but `PermissionError` (EPERM) on macOS.
Teardown paths that suppress only `ProcessLookupError` therefore fail
intermittently on macOS -- this caused a ~1-in-3 flake in the eval isolation
cleanup before it was fixed.

There is no shared process-group helper in this repo; each call site rolls its
own teardown across five packages, so this check lives here rather than beside
any one of them.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKIP_PARTS = {"__pycache__", ".venv", ".git", "node_modules", ".artifacts"}

# Any of these, named on the enclosing guard, absorbs EPERM.
TOLERANT = {"PermissionError", "OSError", "EnvironmentError", "Exception", "BaseException"}


def _python_files() -> list[Path]:
    return [path for path in REPO_ROOT.rglob("*.py") if not SKIP_PARTS & set(path.parts)]


def _exception_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for element in node.elts:
            names |= _exception_names(element)
        return names
    return set()


def _guard_names(node: ast.AST) -> set[str]:
    """Exception names an enclosing `except` or `with suppress(...)` absorbs."""

    names: set[str] = set()
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            names |= _exception_names(handler.type)
            if handler.type is None:  # bare except
                names.add("BaseException")
    elif isinstance(node, ast.With | ast.AsyncWith):
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call):
                target = call.func
                label = (
                    target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                )
                if label == "suppress":
                    for argument in call.args:
                        names |= _exception_names(argument)
    return names


def _is_killpg(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "killpg"
    )


def _unguarded_killpg(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not _is_killpg(node):
            continue
        tolerated: set[str] = set()
        cursor: ast.AST | None = node
        while cursor is not None:
            tolerated |= _guard_names(cursor)
            if isinstance(cursor, ast.FunctionDef | ast.AsyncFunctionDef):
                break
            cursor = parents.get(cursor)
        if not tolerated & TOLERANT:
            offenders.append(getattr(node, "lineno", 0))
    return offenders


def test_every_killpg_tolerates_a_vanished_group() -> None:
    unguarded = {
        str(path.relative_to(REPO_ROOT)): lines
        for path in _python_files()
        if (lines := _unguarded_killpg(path))
    }
    assert not unguarded, (
        "os.killpg must tolerate PermissionError (macOS EPERM on a reaped group), "
        f"but these call sites do not: {unguarded}"
    )


def test_the_check_can_actually_find_an_offender(tmp_path: Path) -> None:
    """Guard against the scan silently matching nothing."""

    offender = tmp_path / "offender.py"
    offender.write_text(
        "import os, signal\n"
        "from contextlib import suppress\n"
        "def stop(pid):\n"
        "    with suppress(ProcessLookupError):\n"
        "        os.killpg(pid, signal.SIGKILL)\n",
        encoding="utf-8",
    )
    assert _unguarded_killpg(offender) == [5]

    offender.write_text(
        "import os, signal\n"
        "from contextlib import suppress\n"
        "def stop(pid):\n"
        "    with suppress(ProcessLookupError, PermissionError):\n"
        "        os.killpg(pid, signal.SIGKILL)\n",
        encoding="utf-8",
    )
    assert _unguarded_killpg(offender) == []
