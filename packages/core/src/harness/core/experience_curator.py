from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.procedures import Procedure, ProcedureLibrary


@dataclass(frozen=True, slots=True)
class CuratorAction:
    kind: str
    procedure_id: str
    name: str
    reason: str
    source_path: Path | None = None
    archive_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CuratorReport:
    scanned: int
    archived: int
    actions: list[CuratorAction] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _procedure_key(procedure: Procedure) -> tuple[str, tuple[str, ...], str]:
    return (
        _normalize_text(procedure.body),
        tuple(trigger.lower() for trigger in procedure.triggers),
        procedure.domain.lower(),
    )


def _archive_target(path: Path) -> Path:
    archive_root = path.parent / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    candidate = archive_root / f"{path.name}-{int(time.time())}"
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"{path.name}-{int(time.time())}-{suffix}"
        suffix += 1
    return candidate


def _archive_procedure(
    procedure: Procedure,
    *,
    reason: str,
    kind: str,
    dry_run: bool,
) -> CuratorAction | None:
    if procedure.path is None:
        return None
    target = _archive_target(procedure.path)
    if not dry_run:
        shutil.move(str(procedure.path), str(target))
    return CuratorAction(
        kind=kind,
        procedure_id=procedure.id,
        name=procedure.name,
        reason=reason,
        source_path=procedure.path,
        archive_path=target,
    )


def curate_procedures(
    paths: list[Path],
    *,
    stale_days: int = 30,
    low_confidence_threshold: float = 1.0,
    dry_run: bool = False,
    now: float | None = None,
) -> CuratorReport:
    library = ProcedureLibrary.load(paths)
    procedures = list(library.procedures)
    actions: list[CuratorAction] = []
    archived_ids: set[str] = set()
    current_time = now if now is not None else time.time()
    stale_cutoff = current_time - (stale_days * 86400)

    grouped: dict[tuple[str, tuple[str, ...], str], list[Procedure]] = {}
    for procedure in procedures:
        grouped.setdefault(_procedure_key(procedure), []).append(procedure)

    for group in grouped.values():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda procedure: (
                procedure.confidence,
                procedure.last_used_at or procedure.created_at,
                procedure.created_at,
            ),
            reverse=True,
        )
        keeper = ordered[0]
        for duplicate in ordered[1:]:
            action = _archive_procedure(
                duplicate,
                reason=f"Exact duplicate of higher-confidence procedure {keeper.id}",
                kind="duplicate",
                dry_run=dry_run,
            )
            if action is not None:
                actions.append(action)
                archived_ids.add(duplicate.id)

    for procedure in procedures:
        if procedure.id in archived_ids:
            continue
        reference_time = procedure.last_used_at or procedure.created_at
        if procedure.confidence < low_confidence_threshold and reference_time < stale_cutoff:
            action = _archive_procedure(
                procedure,
                reason=("Low-confidence procedure is stale and has not been used recently"),
                kind="stale_low_confidence",
                dry_run=dry_run,
            )
            if action is not None:
                actions.append(action)
                archived_ids.add(procedure.id)

    return CuratorReport(scanned=len(procedures), archived=len(actions), actions=actions)


__all__ = ["CuratorAction", "CuratorReport", "curate_procedures"]
