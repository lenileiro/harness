from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.telemetry import get_logger
from harness.core.tips_models import Tip

logger = get_logger("harness.procedures")


def _new_procedure_id() -> str:
    return f"proc_{uuid.uuid4().hex[:10]}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or _new_procedure_id()


@dataclass
class Procedure:
    name: str
    body: str
    triggers: tuple[str, ...] = ()
    domain: str = "general"
    source: str = "human"
    confidence: float = 1.0
    created_from: str | None = None
    id: str = field(default_factory=_new_procedure_id)
    last_used_at: float | None = None
    created_at: float = field(default_factory=time.time)
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path) if self.path else None
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        body: str,
        path: Path | None = None,
    ) -> Procedure:
        triggers = data.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        return cls(
            id=str(data.get("id") or _new_procedure_id()),
            name=str(data.get("name") or "").strip(),
            body=body.strip(),
            triggers=tuple(str(trigger).strip() for trigger in triggers if str(trigger).strip()),
            domain=str(data.get("domain") or "general"),
            source=str(data.get("source") or "human"),
            confidence=float(data.get("confidence", 1.0) or 1.0),
            created_from=(str(data["created_from"]) if data.get("created_from") else None),
            last_used_at=(
                float(data["last_used_at"]) if data.get("last_used_at") is not None else None
            ),
            created_at=float(data.get("created_at", time.time())),
            path=path,
        )

    def matches(self, task_text: str) -> bool:
        if not self.triggers:
            return True
        lowered = task_text.lower()
        return any(trigger.lower() in lowered for trigger in self.triggers)

    def as_tip(self) -> Tip:
        return Tip(
            id=self.id,
            text=self.body,
            triggers=self.triggers,
            weight=self.confidence,
            source_session_id=self.created_from,
            created_at=self.created_at,
        )


@dataclass
class ProcedureLibrary:
    root: Path | None = None
    procedures: list[Procedure] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path]) -> ProcedureLibrary:
        loaded: list[Procedure] = []
        write_root: Path | None = None
        for candidate in paths:
            if write_root is None:
                write_root = candidate
            if not candidate.exists():
                continue
            if candidate.is_file():
                dirs = [candidate.parent]
            else:
                dirs = sorted(path for path in candidate.iterdir() if path.is_dir())
            for directory in dirs:
                meta_path = directory / "procedure.json"
                body_path = directory / "PROCEDURE.md"
                if not meta_path.is_file() or not body_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    body = body_path.read_text(encoding="utf-8")
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("procedures.load_failed", path=str(directory), error=str(exc))
                    continue
                if not isinstance(meta, dict):
                    continue
                loaded.append(Procedure.from_dict(meta, body=body, path=directory))
        return cls(root=write_root, procedures=loaded)

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matched = [procedure for procedure in self.procedures if procedure.matches(task_text)]
        matched.sort(key=lambda procedure: procedure.confidence, reverse=True)
        return [procedure.as_tip() for procedure in matched[:top_k]]

    def add(self, procedure: Procedure) -> Path:
        if self.root is None:
            raise ValueError("procedure library has no writable root")
        target = self.root / _slugify(procedure.name)
        target.mkdir(parents=True, exist_ok=True)
        (target / "procedure.json").write_text(
            json.dumps(
                {
                    "id": procedure.id,
                    "name": procedure.name,
                    "triggers": list(procedure.triggers),
                    "domain": procedure.domain,
                    "source": procedure.source,
                    "confidence": procedure.confidence,
                    "created_from": procedure.created_from,
                    "last_used_at": procedure.last_used_at,
                    "created_at": procedure.created_at,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (target / "PROCEDURE.md").write_text(procedure.body.rstrip() + "\n", encoding="utf-8")
        procedure.path = target
        self.procedures.append(procedure)
        return target

    def __bool__(self) -> bool:
        return bool(self.procedures)


def default_procedure_paths() -> list[Path]:
    return [
        Path.cwd() / ".harness" / "procedures",
        Path.home() / ".harness" / "procedures",
    ]


__all__ = ["Procedure", "ProcedureLibrary", "default_procedure_paths"]
