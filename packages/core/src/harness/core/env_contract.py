"""L1 — Environment contract registry.

Spec borrowed from the LifeHarness paper (Peking U., 2026). Their L1 layer
"calibrates the tool descriptions and the interface constraints" by
intercepting the prompt before the model sees the task and injecting hard
rules (e.g. "avoid recursive broad searches — the env will crash"). In our
setting the rules are project- and tool-specific:

  • "don't pipe untrusted URLs to sh"
  • "the shell tool runs in cwd=<path>; relative paths only"
  • "this repo uses uv; use `uv run` not `python` directly"
  • "fixture has 3 pre-existing noise issues; do not touch them"

The registry loads ``EnvironmentContract`` entries from one or more search
paths (default ``.harness/contracts/*.yaml`` relative to cwd and
``~/.harness/contracts/*.yaml``). At run start, the runtime asks the
registry which contracts apply to the current task text — matching is
keyword-trigger based, with the empty trigger list meaning "always apply."
Matched contracts are joined into a single system message and prepended
to the run.

Loading is best-effort: a malformed file logs a warning and is skipped;
the runtime continues with whatever loaded successfully. Contracts are
inert by themselves — they only matter once the runtime asks for them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.core.telemetry import get_logger

logger = get_logger("harness.env_contract")


@dataclass(frozen=True)
class EnvironmentContract:
    """A single hard rule (or set of rules) for the agent's interface.

    Args:
        name: short identifier, used in activity events and rendering.
        rules: ordered list of imperative one-liners. Each becomes a
            bullet in the injected system message.
        triggers: keyword / regex patterns that match the task text. Any
            match → contract applies. Empty list → always applies.
        regex: when True, ``triggers`` are treated as Python regexes
            (case-insensitive) instead of literal substrings.
        priority: higher numbers win when contracts conflict (the registry
            sorts by priority desc when injecting so the most specific
            rules appear first).
        source: where this contract was loaded from. Filled by the loader.
    """

    name: str
    rules: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    regex: bool = False
    priority: int = 0
    source: str | None = None

    def matches(self, task_text: str) -> bool:
        """True if this contract applies to the given task."""
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


@dataclass
class ContractRegistry:
    """Holds environment contracts and matches them to a task.

    Construct with ``ContractRegistry.from_paths(paths)`` or pass contracts
    directly for tests. The registry is read-only after construction.
    """

    contracts: list[EnvironmentContract] = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths: list[Path] | None = None) -> ContractRegistry:
        """Load contracts from one or more directories of YAML / JSON files.

        Default paths: ``./.harness/contracts/`` and
        ``~/.harness/contracts/``. Files ending in ``.yaml`` / ``.yml`` /
        ``.json`` are read. Anything else is ignored. Bad files log a
        warning and are skipped — never crash the run for a parse error.
        """
        search = paths if paths is not None else _default_paths()
        loaded: list[EnvironmentContract] = []
        for directory in search:
            if not directory.exists() or not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.suffix.lower() not in (".yaml", ".yml", ".json"):
                    continue
                try:
                    loaded.extend(_load_file(entry))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("env_contract.load_failed", path=str(entry), error=str(exc))
        return cls(contracts=loaded)

    def match(self, task_text: str) -> list[EnvironmentContract]:
        """Return contracts that apply to this task, highest priority first."""
        return sorted(
            (c for c in self.contracts if c.matches(task_text)),
            key=lambda c: c.priority,
            reverse=True,
        )

    def render(self, task_text: str) -> str | None:
        """Render matching contracts into a single system-message string.

        Returns None when nothing matches so the runtime can skip the
        injection entirely (no empty messages in the transcript).
        """
        matched = self.match(task_text)
        if not matched:
            return None
        lines: list[str] = ["[harness:L1 environment contracts] hard rules for this run:"]
        for contract in matched:
            lines.append(f"  • {contract.name}:")
            for rule in contract.rules:
                lines.append(f"      - {rule}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return bool(self.contracts)


def _default_paths() -> list[Path]:
    return [Path.cwd() / ".harness" / "contracts", Path.home() / ".harness" / "contracts"]


def _load_file(path: Path) -> list[EnvironmentContract]:
    """Parse a single contract file. JSON or YAML, single contract or list."""
    text = path.read_text(encoding="utf-8")
    payload = _parse_payload(text, path)
    if payload is None:
        return []
    if isinstance(payload, dict):
        items: list[dict] = [payload]
    elif isinstance(payload, list):
        items = [p for p in payload if isinstance(p, dict)]
    else:
        logger.warning("env_contract.bad_shape", path=str(path), got=type(payload).__name__)
        return []
    return [_to_contract(item, source=str(path)) for item in items]


def _parse_payload(text: str, path: Path) -> Any:
    """Try YAML first (PyYAML is a soft dep), fall back to JSON.

    YAML support is optional; if PyYAML isn't installed, ``.yaml`` files
    can still be parsed as JSON when they happen to be valid JSON. This
    keeps the feature usable without forcing a new hard dependency on
    the core package.
    """
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]

            return yaml.safe_load(text)
        except ModuleNotFoundError:
            logger.debug("env_contract.yaml_unavailable_falling_back_to_json")
        except Exception as exc:
            logger.warning("env_contract.yaml_parse_failed", path=str(path), error=str(exc))
            return None
    # JSON fallback (works for `.json` files and YAML-that-is-JSON).
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("env_contract.json_parse_failed", path=str(path), error=str(exc))
        return None


def _to_contract(item: dict, *, source: str) -> EnvironmentContract:
    """Coerce a raw dict from disk into an EnvironmentContract."""
    name = str(item.get("name") or "unnamed").strip()
    rules_raw = item.get("rules") or []
    if isinstance(rules_raw, str):
        rules_raw = [rules_raw]
    rules = tuple(str(r) for r in rules_raw if str(r).strip())
    triggers_raw = item.get("triggers") or []
    if isinstance(triggers_raw, str):
        triggers_raw = [triggers_raw]
    triggers = tuple(str(t) for t in triggers_raw if str(t).strip())
    return EnvironmentContract(
        name=name,
        rules=rules,
        triggers=triggers,
        regex=bool(item.get("regex", False)),
        priority=int(item.get("priority", 0) or 0),
        source=source,
    )


__all__ = ["ContractRegistry", "EnvironmentContract"]
