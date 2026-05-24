"""Tests for the fixture mutator."""

from __future__ import annotations

import sys
from pathlib import Path

# Add evals/ to sys.path so we can import the mutator module by name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mutator import (
    Mutation,
    apply_mutation,
    materialize_fixture_set,
    mutate_fixture,
    plan_mutation,
)


class TestPlanMutation:
    def test_includes_only_symbols_present_in_text(self) -> None:
        text = "class Calculator:\n    def power(self): ...\n"
        plan = plan_mutation(text, seed=1)
        assert "Calculator" in plan.renames
        assert "power" in plan.renames
        assert "Cache" not in plan.renames  # not in text

    def test_deterministic_for_same_seed(self) -> None:
        text = "class Calculator: pass\n"
        a = plan_mutation(text, seed=42)
        b = plan_mutation(text, seed=42)
        assert a.renames == b.renames

    def test_different_seeds_likely_differ(self) -> None:
        # With pool of 3 alternatives, two different seeds usually pick
        # different targets. Not a hard guarantee but holds for at least
        # one of the small-integer seeds.
        text = "class Calculator: pass\n"
        renames_per_seed = {s: plan_mutation(text, seed=s).renames["Calculator"] for s in range(10)}
        # At least two distinct picks across seeds 0..9.
        assert len(set(renames_per_seed.values())) >= 2

    def test_empty_pool_produces_no_renames(self) -> None:
        text = "import os\nx = 1\n"
        plan = plan_mutation(text, seed=1)
        assert plan.renames == {}


class TestApplyMutation:
    def test_renames_python_source(self, tmp_path: Path) -> None:
        src = tmp_path / "src_fixture"
        dest = tmp_path / "dest_fixture"
        src.mkdir()
        (src / "calculator.py").write_text(
            "class Calculator:\n    def power(self, b, e): return b ** e\n"
        )
        (src / "test_calculator.py").write_text("def test_power(): Calculator().power(2, 3)\n")
        mutation = Mutation(seed=1, renames={"Calculator": "MathBox", "power": "exponentiate"})
        touched = apply_mutation(src, dest, mutation)
        assert len(touched) == 2
        result = (dest / "calculator.py").read_text()
        assert "MathBox" in result
        assert "exponentiate" in result
        assert "Calculator" not in result
        assert "power" not in result

    def test_word_boundary_protects_substrings(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "x.py").write_text("class Cache: pass\nclass MyCacheHelper: pass\n")
        mutation = Mutation(seed=1, renames={"Cache": "Store"})
        apply_mutation(src, dest, mutation)
        result = (dest / "x.py").read_text()
        assert "class Store:" in result
        # Substring of CacheHelper must NOT be rewritten
        assert "MyCacheHelper" in result
        assert "MyStoreHelper" not in result

    def test_overwrites_existing_dest(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (dest / "leftover.txt").write_text("stale")
        (src / "fresh.py").write_text("class Calculator: pass\n")
        apply_mutation(src, dest, Mutation(seed=1, renames={"Calculator": "X"}))
        assert (dest / "fresh.py").exists()
        assert not (dest / "leftover.txt").exists()


class TestMutateFixture:
    def test_writes_to_seeded_subdir(self, tmp_path: Path) -> None:
        # Build a minimal fixture
        fixture_dir = tmp_path / "fixtures" / "demo"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "src.py").write_text("class Calculator: ...\n")
        (fixture_dir / "TASK.md").write_text("Use the Calculator class.\n")
        out_root = tmp_path / "fixtures-mutated"
        result = mutate_fixture(fixture_dir, seed=42, dest_root=out_root)
        assert result.dest_dir.name == "0042-demo"
        assert result.dest_dir.parent == out_root
        assert "Calculator" in result.renames

    def test_no_renames_when_no_pool_symbols_present(self, tmp_path: Path) -> None:
        fixture_dir = tmp_path / "fixtures" / "x"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "src.py").write_text("def helper():\n    return 1\n")
        out_root = tmp_path / "out"
        result = mutate_fixture(fixture_dir, seed=1, dest_root=out_root)
        assert result.renames == {}
        # Dest still exists with verbatim copy
        assert (result.dest_dir / "src.py").exists()


class TestMaterializeFixtureSet:
    def test_materializes_mixed_fixture_set(self, tmp_path: Path) -> None:
        fixture_root = tmp_path / "fixtures"
        first = fixture_root / "01-one"
        second = fixture_root / "02-two"
        for path in (first, second):
            path.mkdir(parents=True)
            (path / "TASK.md").write_text("Use Calculator power.\n")
            (path / "EVAL.md").write_text("trap: >\n  avoid drift\n")
            (path / "fixture.yaml").write_text("family: demo\n")
            (path / "src.py").write_text("class Calculator:\n    def power(self): pass\n")

        result = materialize_fixture_set(
            fixture_root,
            dest_root=tmp_path / "generated",
            mode="mixed",
            seeds=[7],
        )

        assert len(result.fixture_dirs) == 2
        assert (result.dest_root / "mutation_manifest.json").exists()
        assert result.mutation_coverage == 0.5
        mutated_yaml = (result.fixture_dirs[1] / "fixture.yaml").read_text()
        assert "mutated_from: 02-two" in mutated_yaml
