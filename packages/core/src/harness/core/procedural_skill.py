"""L2 — Procedural skill (reusable tips mined from prior failure traces).

Spec borrowed from the LifeHarness paper (Peking U., 2026). Their L2 layer
is "a tiny RAG without retrieval of facts" — it retrieves *procedural
skills* (lessons learned from prior failures) for the current task and
injects them into the prompt. The LifeHarness example: after watching a
4B model fail at web-shop, an offline Codex pass extracted the tip
"remember petite and tall sizes are grouped together in this shop" — a
project-specific procedural fact that no amount of reasoning would let
a frozen model rediscover.

Our implementation is split into two pieces:

  Tip                — a single (id, triggers, text, source) record.
  TipLibrary         — file-backed (JSONL) store + keyword/regex matcher.
                       Used at run start: query with task text, get the
                       top-K matching tips, render them as a system block.
  mine_tips_from_run — offline LLM-driven extractor invoked by
                       ``harness tips mine`` (CLI), reads recent failed
                       sessions and asks the model to produce one-sentence
                       tips. Stored as the LifeHarness "frozen patches."

The library file lives at ``~/.harness/tips.jsonl`` by default, or
``.harness/tips.jsonl`` for repo-scoped tips. Each line is a JSON object
matching ``Tip.as_dict()``. We use JSONL (not a single JSON document) so
manual ``harness tips add`` writes are append-only and don't risk
corrupting earlier entries.

The TipsProvider Protocol is what the runtime sees — the runtime doesn't
care whether tips came from a file, a database, or a future RAG service.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from harness.core.telemetry import get_logger

logger = get_logger("harness.procedural_skill")


def _new_id() -> str:
    return f"tip_{uuid.uuid4().hex[:10]}"


@dataclass
class Tip:
    """A single mined-or-authored procedural skill."""

    text: str
    triggers: tuple[str, ...] = ()
    """Substrings or regexes matched against the task text. Empty = always."""
    weight: float = 1.0
    """Higher weights bubble up. Mining can set this from the original
    failure's verifier verdict (e.g., the more severe the verdict, the
    higher the weight)."""
    id: str = field(default_factory=_new_id)
    source_session_id: str | None = None
    """When mined automatically, the session this tip was distilled from."""
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
            triggers=tuple(str(t) for t in triggers if str(t).strip()),
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
        for trig in self.triggers:
            if self.regex:
                if re.search(trig, text, re.IGNORECASE):
                    return True
            else:
                if trig.lower() in text:
                    return True
        return False


@runtime_checkable
class TipsProvider(Protocol):
    """What the runtime sees. Anything that can answer ``query(task) -> tips``."""

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]: ...


@dataclass
class TipLibrary:
    """File-backed Tip store, JSONL on disk.

    Construct with ``TipLibrary.load(paths)`` to read existing tips, or
    plain construction for tests. Writes (``add``, ``save``) are
    append-only and durable.

    Args:
        path: file the library writes new tips to. When the library was
            loaded from multiple paths, ``path`` is the first writable one.
        tips: in-memory list of loaded tips.
    """

    path: Path | None = None
    tips: list[Tip] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> TipLibrary:
        """Load tips from one or more JSONL files. Missing files are skipped."""
        search = paths if paths is not None else _default_tip_paths()
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
        """Return up to ``top_k`` tips ordered by weight desc that match."""
        matches = [t for t in self.tips if t.matches(task_text)]
        matches.sort(key=lambda t: t.weight, reverse=True)
        return matches[:top_k]

    def add(self, tip: Tip, *, persist: bool = True) -> None:
        """Append a tip to memory and optionally to disk.

        Disk writes use append mode so concurrent writes from another
        process do not race the in-memory representation.
        """
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


def _default_tip_paths() -> list[Path]:
    """Where the library looks for tips.jsonl, repo-first then home."""
    return [
        Path.cwd() / ".harness" / "tips.jsonl",
        Path.home() / ".harness" / "tips.jsonl",
    ]


def keywords_from_text(text: str, *, max_keywords: int = 5) -> list[str]:
    """Heuristic trigger extraction for new tips when no explicit triggers given.

    Pulls the longest unique alpha-numeric tokens from ``text`` as a quick
    way to derive triggers from a tip body. Not great, but fine for a
    starting point — humans can edit the triggers after mining.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text)
    seen: dict[str, None] = {}
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        if key in _STOPWORDS:
            continue
        seen[key] = None
        if len(seen) >= max_keywords:
            break
    return list(seen.keys())


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


# ---------------------------------------------------------------------------
# Offline mining — used by `harness tips mine`
# ---------------------------------------------------------------------------


@dataclass
class MiningInput:
    """Bundle handed to the LLM extractor for a single failed session.

    Build one per session you want to mine. The extractor returns 0..N
    Tip objects.
    """

    session_id: str
    task_text: str
    failure_summary: str
    """Short narrative — verifier verdict + key tool failures."""
    transcript_excerpt: str
    """Tail of the transcript (~last 2k chars); full history is overkill."""


def render_mining_prompt(item: MiningInput) -> str:
    """The prompt sent to the mining LLM. Stable shape for reproducibility."""
    return (
        "You are a harness-improvement extractor. Read the failed agent "
        "session below and extract 0 to 3 short, *procedural* tips that would "
        "have prevented the failure. Each tip should be a single imperative "
        "sentence under 20 words. Do not paraphrase the task — extract a "
        "*generalizable lesson* about how to interact with this codebase or "
        "tooling.\n"
        f"\n--- TASK ---\n{item.task_text.strip()}\n"
        f"\n--- FAILURE ---\n{item.failure_summary.strip()}\n"
        f"\n--- TRANSCRIPT (tail) ---\n{item.transcript_excerpt.strip()}\n"
        "\nReturn ONLY valid JSON in this shape:\n"
        '  {"tips": [{"text": "...", "triggers": ["keyword", "..."]}]}\n'
        'Return `{"tips": []}` when no useful lesson can be extracted.'
    )


def parse_mined_tips(response: str, *, source_session_id: str | None = None) -> list[Tip]:
    """Coerce the mining-LLM's JSON response into Tip objects."""
    body = response.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("tips.mining_bad_json", preview=body[:200])
        return []
    raw = payload.get("tips") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[Tip] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or len(text) > 200:
            continue
        triggers = item.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        triggers = [str(t).strip() for t in triggers if str(t).strip()]
        out.append(
            Tip(
                text=text,
                triggers=tuple(triggers),
                source_session_id=source_session_id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Bring-your-own provider helper
# ---------------------------------------------------------------------------


@dataclass
class StaticTipsProvider:
    """Pass an explicit list of tips. Useful for tests and inline configs."""

    tips: Iterable[Tip]

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matches = [t for t in self.tips if t.matches(task_text)]
        matches.sort(key=lambda t: t.weight, reverse=True)
        return matches[:top_k]


__all__ = [
    "MiningInput",
    "StaticTipsProvider",
    "Tip",
    "TipLibrary",
    "TipsProvider",
    "keywords_from_text",
    "parse_mined_tips",
    "render_mining_prompt",
]
