from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.extensions import ExperienceProvider
from harness.core.procedures import ProcedureLibrary, default_procedure_paths
from harness.core.tips_models import (
    Tip,
    TipLibrary,
    default_experience_paths,
    default_tip_paths,
    logger,
)


@dataclass
class ArtifactExperienceProvider:
    tips: list[Tip] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> ArtifactExperienceProvider:
        search = paths if paths is not None else default_experience_paths()
        loaded: list[Tip] = []
        seen_ids: set[str] = set()
        for candidate in search:
            if candidate.is_file():
                files = [candidate] if candidate.name == "harness_adjustments.json" else []
            elif candidate.is_dir():
                files = sorted(candidate.rglob("harness_adjustments.json"))
            else:
                continue
            for file_path in files:
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "artifact_experience.load_failed",
                        path=str(file_path),
                        error=str(exc),
                    )
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
class CompositeExperienceProvider:
    providers: list[ExperienceProvider]

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        merged: list[Tip] = []
        seen_ids: set[str] = set()
        for provider in self.providers:
            try:
                matches = provider.query(task_text, top_k=top_k)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("experience.provider_failed", error=str(exc))
                continue
            for tip in matches:
                if tip.id in seen_ids:
                    continue
                merged.append(tip)
                seen_ids.add(tip.id)
        merged.sort(key=lambda tip: tip.weight, reverse=True)
        return merged[:top_k]


@dataclass
class StaticExperienceProvider:
    tips: Iterable[Tip]

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
        matches = [tip for tip in self.tips if tip.matches(task_text)]
        matches.sort(key=lambda tip: tip.weight, reverse=True)
        return matches[:top_k]


def default_experience_roots(cwd: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    roots = os.environ.get("HARNESS_EXPERIENCE_ROOTS", "")
    for raw in roots.split(os.pathsep):
        raw = raw.strip()
        if raw:
            paths.append(Path(raw))
    repo_runs = (cwd or Path.cwd()) / "evals" / "runs"
    if repo_runs not in paths:
        paths.append(repo_runs)
    return paths


def load_default_experience_provider(
    *,
    cwd: Path | None = None,
    tip_paths: list[Path] | None = None,
    artifact_paths: list[Path] | None = None,
    procedure_paths: list[Path] | None = None,
    extra_providers: list[ExperienceProvider] | None = None,
) -> ExperienceProvider | None:
    static_library = TipLibrary.load(tip_paths or default_tip_paths())
    procedure_library = ProcedureLibrary.load(procedure_paths or default_procedure_paths())
    artifact_provider = ArtifactExperienceProvider.load(
        artifact_paths or default_experience_roots(cwd)
    )
    providers: list[ExperienceProvider] = []
    if static_library:
        providers.append(static_library)
    if procedure_library:
        providers.append(procedure_library)
    if artifact_provider:
        providers.append(artifact_provider)
    if extra_providers:
        providers.extend(extra_providers)
    if len(providers) == 1:
        return providers[0]
    if providers:
        return CompositeExperienceProvider(providers=providers)
    return None


__all__ = [
    "ArtifactExperienceProvider",
    "CompositeExperienceProvider",
    "ExperienceProvider",
    "StaticExperienceProvider",
    "default_experience_roots",
    "load_default_experience_provider",
]
