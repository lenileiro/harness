from __future__ import annotations

from pathlib import Path

from harness.core.procedures import Procedure, ProcedureLibrary


def test_procedure_library_add_writes_artifact_files(tmp_path: Path) -> None:
    library = ProcedureLibrary(root=tmp_path / "procedures")
    procedure = Procedure(
        name="Minimal null-guard fix",
        body="Add the null guard in the named formatter and avoid sibling cleanup.",
        triggers=("format_price", "null"),
        domain="coding",
        source="human",
        confidence=2.0,
    )

    target = library.add(procedure)

    assert (target / "procedure.json").is_file()
    assert (target / "PROCEDURE.md").read_text(encoding="utf-8").strip() == procedure.body


def test_procedure_library_loads_and_queries_as_experience(tmp_path: Path) -> None:
    library = ProcedureLibrary(root=tmp_path / "procedures")
    library.add(
        Procedure(
            name="Scope discipline",
            body="Keep the patch in the named formatter and avoid broader refactors.",
            triggers=("format_price", "minimal fix"),
            confidence=2.5,
        )
    )

    loaded = ProcedureLibrary.load([tmp_path / "procedures"])
    matches = loaded.query("Need a minimal fix in format_price", top_k=5)

    assert [tip.text for tip in matches] == [
        "Keep the patch in the named formatter and avoid broader refactors."
    ]
