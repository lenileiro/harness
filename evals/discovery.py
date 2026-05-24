"""Fixture discovery and metadata parsing for evals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from evals.types import FixtureMeta, FixtureRules

_DEFAULT_VERIFY_COMMAND = "pytest tests/ -v --tb=short --no-header"


def find_evals_root() -> Path:
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
    root = evals_root or find_evals_root()
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
        metadata = load_fixture_config(entry / "fixture.yaml")
        eval_md = eval_path.read_text(encoding="utf-8")
        phases = coerce_optional_list(metadata.get("phases"))
        if not phases:
            phases = parse_phases(eval_md)
        family = str(metadata.get("family") or entry.name.split("-", 1)[-1])
        rules = rules_from_metadata(eval_md, metadata)
        fixture = FixtureMeta(
            name=entry.name,
            path=entry,
            task_text=task_path.read_text(encoding="utf-8"),
            eval_md=eval_md,
            verify_command=str(metadata.get("verify_command") or _DEFAULT_VERIFY_COMMAND),
            phases=phases,
            family=family,
            holdout=bool(metadata.get("holdout", False)),
            mutated_from=coerce_optional_str(metadata.get("mutated_from")),
            metadata_path=entry / "fixture.yaml" if (entry / "fixture.yaml").exists() else None,
            rules=rules,
        )
        if fixture.holdout and not include_holdout:
            continue
        result.append(fixture)
    return result


def rules_from_metadata(eval_md: str, metadata: dict[str, object]) -> FixtureRules:
    return FixtureRules(
        behavior_category=str(metadata.get("behavior_category") or metadata.get("family") or ""),
        primary_dimension=str(
            metadata.get("primary_dimension") or extract_eval_field(eval_md, "primary_dimension")
        ),
        expected_first_step=str(
            metadata.get("expected_first_step") or expected_first_step_from_eval(eval_md)
        ),
        allowed_paths=ensure_list(metadata.get("allowed_paths")),
        disallowed_paths=ensure_list(metadata.get("disallowed_paths")),
        required_verification=str(metadata.get("required_verification") or ""),
        trap=str(metadata.get("trap") or extract_eval_field(eval_md, "trap")),
        correct_fix=str(metadata.get("correct_fix") or extract_eval_field(eval_md, "correct_fix")),
        dimensions=ensure_list(metadata.get("dimensions"))
        or parse_csv_field(extract_eval_field(eval_md, "dimensions")),
        scoring_notes=str(
            metadata.get("scoring_notes") or extract_eval_field(eval_md, "scoring_notes")
        ),
    )


def ensure_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            return parse_csv_field(inner)
        return parse_csv_field(value)
    return [str(value)]


def coerce_optional_list(value: object) -> list[str] | None:
    items = ensure_list(value)
    return items or None


def coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_csv_field(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def extract_eval_field(eval_md: str, field: str) -> str:
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


def expected_first_step_from_eval(eval_md: str) -> str:
    lower = eval_md.lower()
    if "run the tests" in lower or "ran tests" in lower:
        return "run tests"
    if "inspect" in lower:
        return "inspect code"
    return ""


def load_fixture_config(path: Path) -> dict[str, object]:
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
            config[key] = parse_csv_field(value[1:-1])
        else:
            config[key] = value
    return config


def parse_phases(eval_md: str) -> list[str] | None:
    """Parse a `phases:` line or block from EVAL.md."""
    lines = eval_md.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*phases\s*:\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        inline = m.group(1).strip()
        if inline:
            parts = [p.strip().lower() for p in inline.split(",") if p.strip()]
            return parts or None
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
