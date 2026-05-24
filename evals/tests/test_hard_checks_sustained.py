"""Tests for sustained-coherence hard checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner


def test_sustained_coherence_behavioral_check_rejects_cleanup_scope_creep(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text("- `multiply(a, b)`\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            "import math\n\n"
            "class Calculator:\n"
            "    def power(self, base, exponent):\n"
            "        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "def test_power():\n    assert True\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text("- `power`\n", encoding="utf-8")

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is False
    assert "unused import cleanup landed" in message
    assert "docstring typo was fixed" in message
    assert "comment-style cleanup changed" in message


def test_sustained_coherence_behavioral_check_rejects_new_test_comment_scaffolding(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text("- `multiply(a, b)`\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n\n'
            "    def power(self, base, exponent):\n        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        (
            "# -- add\n"
            "# subtract\n"
            "# -- multiply\n"
            "# sqrt\n"
            "# -- power ---------------------------------------------------------------\n"
            "def test_power(calc):\n"
            "    assert calc.power(2, 3) == 8\n"
            "    assert calc.power(5, 0) == 1\n"
            "    assert calc.power(2, -1) == 0.5\n\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n- `power(base, exponent)`\n",
        encoding="utf-8",
    )

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is False
    assert "new comment-style test scaffolding was added" in message


def test_sustained_coherence_behavioral_check_accepts_minimal_scope(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            '"""Simple calculator."""\n\n'
            "import json  # noqa: F401\n"
            "import math\n\n"
            "class Calculator:\n"
            '    """A stateless calculator."""\n\n'
            '    def multiply(self, a, b):\n        """Multipy `a` and `b`."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text("- `multiply(a, b)`\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            '"""Simple calculator."""\n\n'
            "import json  # noqa: F401\n"
            "import math\n\n"
            "class Calculator:\n"
            '    """A stateless calculator."""\n\n'
            '    def multiply(self, a, b):\n        """Multipy `a` and `b`."""\n        return a * b\n\n'
            "    def power(self, base, exponent):\n        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\ndef test_power():\n    assert True\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text("- `multiply(a, b)`\n- `power`\n", encoding="utf-8")

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is True
    assert "scope preserved" in message
