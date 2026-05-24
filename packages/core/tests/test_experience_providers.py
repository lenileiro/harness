from __future__ import annotations

import json
from pathlib import Path

from harness.core.experience import (
    ArtifactExperienceProvider,
    CompositeExperienceProvider,
    StaticExperienceProvider,
    load_default_experience_provider,
)
from harness.core.tips_models import Tip


def test_artifact_experience_provider_loads_single_json_file(tmp_path: Path) -> None:
    artifact_file = tmp_path / "harness_adjustments.json"
    artifact_file.write_text(
        json.dumps(
            [
                {
                    "id": "adj_1",
                    "text": "rerun the narrow failing test after each repair",
                    "triggers": ["pytest", "repair"],
                    "weight": 2.0,
                    "source_artifact_dir": str(tmp_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    provider = ArtifactExperienceProvider.load([artifact_file])
    assert [tip.text for tip in provider.query("pytest repair loop", top_k=5)] == [
        "rerun the narrow failing test after each repair"
    ]


def test_load_default_experience_provider_merges_static_and_artifacts(tmp_path: Path) -> None:
    tip_path = tmp_path / "tips.jsonl"
    tip_path.write_text(
        json.dumps(Tip(text="prefer uv run", triggers=("uv",), weight=1.0).as_dict()) + "\n",
        encoding="utf-8",
    )

    artifact_dir = tmp_path / "evals" / "runs" / "demo" / "run-01"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "harness_adjustments.json").write_text(
        json.dumps(
            [
                {
                    "id": "adj_2",
                    "text": "run the smallest targeted pytest command first",
                    "triggers": ["pytest"],
                    "weight": 2.5,
                    "source_artifact_dir": str(artifact_dir),
                }
            ]
        ),
        encoding="utf-8",
    )

    provider = load_default_experience_provider(
        cwd=tmp_path,
        tip_paths=[tip_path],
        artifact_paths=[tmp_path / "evals" / "runs"],
    )
    assert isinstance(provider, CompositeExperienceProvider)
    result = provider.query("use uv run and then pytest", top_k=5)
    assert [tip.text for tip in result] == [
        "run the smallest targeted pytest command first",
        "prefer uv run",
    ]


def test_load_default_experience_provider_includes_procedures(tmp_path: Path) -> None:
    procedures = tmp_path / "procedures"
    procedures.mkdir()
    procedure_dir = procedures / "scope-discipline"
    procedure_dir.mkdir()
    (procedure_dir / "procedure.json").write_text(
        """
        {
          "id": "proc_1",
          "name": "Scope discipline",
          "triggers": ["format_price"],
          "domain": "coding",
          "source": "human",
          "confidence": 2.2
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    (procedure_dir / "PROCEDURE.md").write_text(
        "Keep the patch in the named formatter only.\n",
        encoding="utf-8",
    )

    provider = load_default_experience_provider(
        cwd=tmp_path,
        tip_paths=[tmp_path / "missing.jsonl"],
        artifact_paths=[tmp_path / "missing-runs"],
        procedure_paths=[procedures],
    )
    assert provider is not None
    assert [tip.text for tip in provider.query("fix format_price only", top_k=5)] == [
        "Keep the patch in the named formatter only."
    ]


def test_load_default_experience_provider_returns_none_when_empty(tmp_path: Path) -> None:
    provider = load_default_experience_provider(
        cwd=tmp_path,
        tip_paths=[tmp_path / "missing.jsonl"],
        artifact_paths=[tmp_path / "missing-runs"],
    )
    assert provider is None


def test_static_experience_provider_sorts_by_weight() -> None:
    provider = StaticExperienceProvider(tips=[Tip(text="a", weight=1.0), Tip(text="b", weight=3.0)])
    assert [tip.text for tip in provider.query("anything", top_k=10)] == ["b", "a"]


def test_load_default_experience_provider_includes_extra_providers(tmp_path: Path) -> None:
    class DemoProvider:
        def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]:
            return [Tip(text="plugin guidance", triggers=("pytest",), weight=4.0)]

    provider = load_default_experience_provider(
        cwd=tmp_path,
        tip_paths=[tmp_path / "missing.jsonl"],
        artifact_paths=[tmp_path / "missing-runs"],
        extra_providers=[DemoProvider()],
    )
    assert provider is not None
    assert [tip.text for tip in provider.query("pytest", top_k=5)] == ["plugin guidance"]
