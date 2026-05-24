from __future__ import annotations

from pathlib import Path

from harness.core import Procedure, ProcedureLibrary, curate_procedures


def _seed_duplicate_procedures(root: Path) -> None:
    library = ProcedureLibrary(root=root)
    library.add(
        Procedure(
            id="proc_keep",
            name="Scope discipline",
            body="Keep the patch in the named formatter only.",
            triggers=("format_price",),
            confidence=2.0,
            created_at=100.0,
        )
    )
    library.add(
        Procedure(
            id="proc_dup",
            name="Scope discipline duplicate",
            body="Keep the patch in the named formatter only.",
            triggers=("format_price",),
            confidence=1.0,
            created_at=90.0,
        )
    )


def test_curate_procedures_archives_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "procedures"
    _seed_duplicate_procedures(root)

    report = curate_procedures([root], dry_run=False, now=10_000.0)

    assert report.scanned == 2
    assert report.archived == 1
    assert report.actions[0].kind == "duplicate"
    assert (root / ".archive").is_dir()


def test_curate_procedures_archives_stale_low_confidence(tmp_path: Path) -> None:
    root = tmp_path / "procedures"
    library = ProcedureLibrary(root=root)
    library.add(
        Procedure(
            id="proc_old",
            name="Old low-confidence",
            body="Try a broad cleanup.",
            confidence=0.5,
            created_at=0.0,
            last_used_at=0.0,
        )
    )

    report = curate_procedures(
        [root],
        stale_days=30,
        low_confidence_threshold=1.0,
        dry_run=False,
        now=60 * 86400.0,
    )

    assert report.archived == 1
    assert report.actions[0].kind == "stale_low_confidence"


def test_curate_procedures_dry_run_does_not_move_files(tmp_path: Path) -> None:
    root = tmp_path / "procedures"
    _seed_duplicate_procedures(root)

    report = curate_procedures([root], dry_run=True, now=10_000.0)

    assert report.archived == 1
    assert any(path.name.startswith("scope-discipline") for path in root.iterdir())
