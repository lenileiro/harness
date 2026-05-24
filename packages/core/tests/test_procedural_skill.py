"""Tests for L2 — procedural skill / tips library."""

from __future__ import annotations

import json
from pathlib import Path

from harness.core import (
    ArtifactTipProvider,
    CompositeTipsProvider,
    StaticTipsProvider,
    Tip,
    TipLibrary,
    parse_mined_tips,
    render_mining_prompt,
)
from harness.core.procedural_skill import MiningInput


class TestTipMatching:
    def test_empty_triggers_always_match(self) -> None:
        tip = Tip(text="always do X")
        assert tip.matches("anything")
        assert tip.matches("")

    def test_substring_trigger_case_insensitive(self) -> None:
        tip = Tip(text="use uv run", triggers=("uv",))
        assert tip.matches("run via UV")
        assert not tip.matches("run via pip")

    def test_regex_trigger(self) -> None:
        tip = Tip(text="be careful", triggers=(r"npm\s+publish",), regex=True)
        assert tip.matches("we npm  publish here")
        assert not tip.matches("npm install only")


class TestTipLibraryQuery:
    def test_top_k_orders_by_weight(self) -> None:
        lib = TipLibrary(
            tips=[
                Tip(text="low", weight=1.0),
                Tip(text="high", weight=10.0),
                Tip(text="mid", weight=5.0),
            ]
        )
        result = lib.query("anything", top_k=2)
        assert [t.text for t in result] == ["high", "mid"]

    def test_query_filters_by_trigger(self) -> None:
        lib = TipLibrary(
            tips=[
                Tip(text="curl", triggers=("curl",), weight=10),
                Tip(text="git", triggers=("git",), weight=5),
            ]
        )
        result = lib.query("we use curl here")
        assert [t.text for t in result] == ["curl"]


class TestTipLibraryPersistence:
    def test_add_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "tips.jsonl"
        lib = TipLibrary(path=path)
        lib.add(Tip(text="hello world", triggers=("hi",)))
        lib.add(Tip(text="another tip"))
        data = path.read_text().strip().splitlines()
        assert len(data) == 2
        first = json.loads(data[0])
        assert first["text"] == "hello world"
        assert first["triggers"] == ["hi"]

    def test_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "tips.jsonl"
        # Hand-craft a JSONL file.
        path.write_text(
            json.dumps(Tip(text="t1", triggers=("trig",)).as_dict())
            + "\n"
            + json.dumps(Tip(text="t2").as_dict())
            + "\n"
        )
        lib = TipLibrary.load([path])
        assert len(lib.tips) == 2
        assert {t.text for t in lib.tips} == {"t1", "t2"}

    def test_load_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "tips.jsonl"
        path.write_text("# header comment\n\n" + json.dumps(Tip(text="real").as_dict()) + "\n")
        lib = TipLibrary.load([path])
        assert len(lib.tips) == 1
        assert lib.tips[0].text == "real"

    def test_missing_file_returns_empty_library(self, tmp_path: Path) -> None:
        lib = TipLibrary.load([tmp_path / "nope.jsonl"])
        assert lib.tips == []
        assert not lib


class TestStaticTipsProvider:
    def test_query_filters_and_sorts(self) -> None:
        provider = StaticTipsProvider(
            tips=[
                Tip(text="a", weight=1.0),
                Tip(text="b", weight=2.0),
            ]
        )
        result = provider.query("anything", top_k=10)
        assert [t.text for t in result] == ["b", "a"]


class TestArtifactTipProvider:
    def test_loads_adjustments_from_artifact_json(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "evals" / "runs" / "demo" / "run-01"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "harness_adjustments.json").write_text(
            json.dumps(
                [
                    {
                        "id": "adj_1",
                        "text": "run tests before editing",
                        "triggers": ["pytest", "format_price"],
                        "weight": 2.5,
                        "source_artifact_dir": str(artifact_dir),
                    }
                ]
            ),
            encoding="utf-8",
        )

        provider = ArtifactTipProvider.load([tmp_path / "evals" / "runs"])

        result = provider.query("pytest format_price failure", top_k=5)
        assert [tip.text for tip in result] == ["run tests before editing"]


class TestCompositeTipsProvider:
    def test_merges_library_and_artifact_sources(self) -> None:
        provider = CompositeTipsProvider(
            providers=[
                StaticTipsProvider(tips=[Tip(text="a", triggers=("curl",), weight=1.0)]),
                StaticTipsProvider(tips=[Tip(text="b", triggers=("curl",), weight=3.0)]),
            ]
        )

        result = provider.query("curl this", top_k=5)

        assert [tip.text for tip in result] == ["b", "a"]


class TestMining:
    def test_render_mining_prompt_contains_task_and_failure(self) -> None:
        inp = MiningInput(
            session_id="s1",
            task_text="fix the bug",
            failure_summary="tests failed",
            transcript_excerpt="[user] do it\n[assistant] done",
        )
        prompt = render_mining_prompt(inp)
        assert "fix the bug" in prompt
        assert "tests failed" in prompt
        assert "do it" in prompt
        assert "JSON" in prompt

    def test_parse_mined_tips_well_formed(self) -> None:
        body = json.dumps(
            {
                "tips": [
                    {"text": "always run pytest first", "triggers": ["test"]},
                    {"text": "respect file scope"},
                ]
            }
        )
        tips = parse_mined_tips(body, source_session_id="s99")
        assert len(tips) == 2
        assert tips[0].text == "always run pytest first"
        assert tips[0].triggers == ("test",)
        assert tips[0].source_session_id == "s99"

    def test_parse_mined_tips_strips_markdown_fence(self) -> None:
        body = "```json\n" + json.dumps({"tips": [{"text": "hi"}]}) + "\n```"
        tips = parse_mined_tips(body)
        assert len(tips) == 1 and tips[0].text == "hi"

    def test_parse_mined_tips_rejects_overlong(self) -> None:
        long_text = "x" * 500
        body = json.dumps({"tips": [{"text": long_text}, {"text": "short"}]})
        tips = parse_mined_tips(body)
        assert [t.text for t in tips] == ["short"]

    def test_parse_mined_tips_bad_json_returns_empty(self) -> None:
        tips = parse_mined_tips("{not json")
        assert tips == []

    def test_parse_mined_tips_missing_tips_key(self) -> None:
        tips = parse_mined_tips(json.dumps({"other": "value"}))
        assert tips == []
