"""Tests for the cross-session ResumeContract."""

from __future__ import annotations

import json
from pathlib import Path

from harness.core import FeatureItem, ResumeContract


class TestFeatureItem:
    def test_roundtrip(self) -> None:
        f = FeatureItem(name="x", description="d", phases=["a", "b"], notes=["n"])
        assert FeatureItem.from_dict(f.as_dict()) == f

    def test_missing_fields_use_defaults(self) -> None:
        f = FeatureItem.from_dict({"name": "x"})
        assert f.name == "x"
        assert f.status == "pending"
        assert f.phases == []


class TestResumeContractLookup:
    def test_current_feature_returns_named_entry(self) -> None:
        rc = ResumeContract(
            current="b",
            features=[FeatureItem(name="a"), FeatureItem(name="b", description="hit")],
        )
        cur = rc.current_feature()
        assert cur is not None and cur.description == "hit"

    def test_current_feature_none_when_current_unset(self) -> None:
        rc = ResumeContract(features=[FeatureItem(name="a")])
        assert rc.current_feature() is None


class TestRender:
    def test_no_current_returns_none(self) -> None:
        rc = ResumeContract()
        assert rc.render_for_prompt() is None

    def test_render_includes_current_feature(self) -> None:
        rc = ResumeContract(
            current="add-foo",
            features=[FeatureItem(name="add-foo", description="ship foo", status="in_progress")],
        )
        rendered = rc.render_for_prompt()
        assert rendered is not None
        assert "add-foo" in rendered
        assert "ship foo" in rendered
        assert "in_progress" in rendered

    def test_render_includes_roadmap_of_others(self) -> None:
        rc = ResumeContract(
            current="a",
            features=[
                FeatureItem(name="a"),
                FeatureItem(name="b", status="pending"),
                FeatureItem(name="c", status="done"),
            ],
        )
        rendered = rc.render_for_prompt()
        assert rendered is not None and "b(pending)" in rendered and "c(done)" in rendered

    def test_render_caps_notes(self) -> None:
        rc = ResumeContract(
            current="a",
            features=[FeatureItem(name="a", notes=[f"note-{i}" for i in range(10)])],
        )
        rendered = rc.render_for_prompt()
        assert rendered is not None
        # Only last 5 notes should appear.
        assert "note-9" in rendered
        assert "note-0" not in rendered


class TestDiskIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "resume.json"
        rc = ResumeContract(current="x", features=[FeatureItem(name="x", description="d")])
        rc.save(target)
        loaded = ResumeContract.load(target)
        assert loaded is not None
        assert loaded.current == "x"
        feat = loaded.current_feature()
        assert feat is not None
        assert feat.description == "d"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert ResumeContract.load(tmp_path / "missing.json") is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("{not json")
        assert ResumeContract.load(target) is None

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "resume.json"
        rc = ResumeContract(current="x", features=[FeatureItem(name="x")])
        rc.save(target)
        assert target.exists()
        assert json.loads(target.read_text())["current"] == "x"
