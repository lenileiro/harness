from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from harness.core.tips_models import Tip, default_experience_paths, logger


@runtime_checkable
class TipsProvider(Protocol):
    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]: ...


@dataclass
class ArtifactTipProvider:
    tips: list[Tip] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> ArtifactTipProvider:
        search = paths if paths is not None else default_experience_paths()
        loaded: list[Tip] = []
        seen_ids: set[str] = set()
        for candidate in search:
            if candidate.is_file():
                files = [candidate]
            elif candidate.is_dir():
                files = sorted(candidate.rglob("harness_adjustments.json"))
            else:
                continue
            for file_path in files:
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("artifact_tips.load_failed", path=str(file_path), error=str(exc))
                    continue
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    tip_id = str(item.get("id") or f"artifact:{file_path}:{text}")
                    if tip_id in seen_ids:
                        continue
                    triggers = item.get("triggers") or []
                    if isinstance(triggers, str):
                        triggers = [triggers]
                    loaded.append(
                        Tip(
                            id=tip_id,
                            text=text,
                            triggers=tuple(
                                str(trigger).strip() for trigger in triggers if str(trigger).strip()
                            ),
                            weight=float(item.get("weight", 1.0) or 1.0),
                            source_session_id=(
                                str(item.get("source_artifact_dir"))
                                if item.get("source_artifact_dir")
                                else None
                            ),
                            created_at=float(item.get("created_at", time.time())),
                        )
                    )
                    seen_ids.add(tip_id)
        return cls(tips=loaded)

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matches = [tip for tip in self.tips if tip.matches(task_text)]
        matches.sort(key=lambda tip: tip.weight, reverse=True)
        return matches[:top_k]

    def __bool__(self) -> bool:
        return bool(self.tips)


@dataclass
class CompositeTipsProvider:
    providers: list[TipsProvider]

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        merged: list[Tip] = []
        seen_ids: set[str] = set()
        for provider in self.providers:
            try:
                matches = provider.query(task_text, top_k=top_k)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("tips.provider_failed", error=str(exc))
                continue
            for tip in matches:
                if tip.id in seen_ids:
                    continue
                merged.append(tip)
                seen_ids.add(tip.id)
        merged.sort(key=lambda tip: tip.weight, reverse=True)
        return merged[:top_k]


@dataclass
class StaticTipsProvider:
    tips: Iterable[Tip]

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matches = [tip for tip in self.tips if tip.matches(task_text)]
        matches.sort(key=lambda tip: tip.weight, reverse=True)
        return matches[:top_k]


__all__ = [
    "ArtifactTipProvider",
    "CompositeTipsProvider",
    "StaticTipsProvider",
    "TipsProvider",
]
