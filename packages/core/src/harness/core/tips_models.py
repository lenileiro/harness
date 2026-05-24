from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.telemetry import get_logger

logger = get_logger("harness.procedural_skill")


def _new_id() -> str:
    return f"tip_{uuid.uuid4().hex[:10]}"


@dataclass
class Tip:
    """A single mined-or-authored procedural skill."""

    text: str
    triggers: tuple[str, ...] = ()
    weight: float = 1.0
    id: str = field(default_factory=_new_id)
    source_session_id: str | None = None
    regex: bool = False
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tip:
        triggers = data.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        return cls(
            text=str(data.get("text") or "").strip(),
            triggers=tuple(str(trigger) for trigger in triggers if str(trigger).strip()),
            weight=float(data.get("weight", 1.0) or 1.0),
            id=str(data.get("id") or _new_id()),
            source_session_id=data.get("source_session_id"),
            regex=bool(data.get("regex", False)),
            created_at=float(data.get("created_at", time.time())),
        )

    def matches(self, task_text: str) -> bool:
        if not self.triggers:
            return True
        text = task_text if self.regex else task_text.lower()
        for trigger in self.triggers:
            if self.regex:
                if re.search(trigger, text, re.IGNORECASE):
                    return True
            elif trigger.lower() in text:
                return True
        return False


@dataclass
class TipLibrary:
    """File-backed Tip store, JSONL on disk."""

    path: Path | None = None
    tips: list[Tip] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> TipLibrary:
        search = paths if paths is not None else default_tip_paths()
        loaded: list[Tip] = []
        write_target: Path | None = None
        for candidate in search:
            if write_target is None:
                write_target = candidate
            if not candidate.exists():
                continue
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        loaded.append(Tip.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.warning("tips.line_skipped", path=str(candidate), error=str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("tips.load_failed", path=str(candidate), error=str(exc))
        return cls(path=write_target, tips=loaded)

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matches = [tip for tip in self.tips if tip.matches(task_text)]
        matches.sort(key=lambda tip: tip.weight, reverse=True)
        return matches[:top_k]

    def add(self, tip: Tip, *, persist: bool = True) -> None:
        self.tips.append(tip)
        if persist and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(tip.as_dict()) + "\n")

    def render(self, task_text: str, *, top_k: int = 3) -> str | None:
        matched = self.query(task_text, top_k=top_k)
        if not matched:
            return None
        lines = ["[harness:L2 procedural tips] lessons from prior runs:"]
        for tip in matched:
            lines.append(f"  • {tip.text}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return bool(self.tips)


def default_tip_paths() -> list[Path]:
    return [
        Path.cwd() / ".harness" / "tips.jsonl",
        Path.home() / ".harness" / "tips.jsonl",
    ]


def default_experience_paths() -> list[Path]:
    paths: list[Path] = []
    roots = os.environ.get("HARNESS_EXPERIENCE_ROOTS", "")
    for raw in roots.split(os.pathsep):
        raw = raw.strip()
        if raw:
            paths.append(Path(raw))
    repo_runs = Path.cwd() / "evals" / "runs"
    if repo_runs not in paths:
        paths.append(repo_runs)
    return paths


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "your",
        "their",
        "they",
        "them",
        "then",
        "than",
        "when",
        "what",
        "where",
        "into",
        "while",
        "should",
        "could",
        "would",
        "tool",
        "tools",
        "agent",
        "harness",
        "test",
        "tests",
        "code",
        "file",
        "files",
    }
)


def keywords_from_text(text: str, *, max_keywords: int = 5) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text)
    seen: dict[str, None] = {}
    for token in tokens:
        key = token.lower()
        if key in seen or key in _STOPWORDS:
            continue
        seen[key] = None
        if len(seen) >= max_keywords:
            break
    return list(seen.keys())


__all__ = [
    "Tip",
    "TipLibrary",
    "default_experience_paths",
    "default_tip_paths",
    "keywords_from_text",
    "logger",
]
