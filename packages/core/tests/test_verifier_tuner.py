"""Tests for the ACON-style verifier prompt tuner."""

from __future__ import annotations

import json
from pathlib import Path

from harness.core.verifier_tuner import (
    ProposedDelta,
    TrajectoryPair,
    TunablePrompt,
    TuneRequest,
    parse_proposal,
    render_tune_prompt,
)


class TestTunablePromptVersioning:
    def test_add_version_increments(self) -> None:
        tp = TunablePrompt(key="x")
        v1 = tp.add_version("seed", rationale="initial")
        v2 = tp.add_version("rev", rationale="tuned")
        assert v1.version == 1 and v2.version == 2
        assert tp.current is not None and tp.current.text == "rev"

    def test_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        tp = TunablePrompt(key="x")
        tp.add_version("seed")
        tp.add_version("rev")
        tp.save(target)
        loaded = TunablePrompt.load(target)
        assert loaded is not None
        assert [v.text for v in loaded.versions] == ["seed", "rev"]

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert TunablePrompt.load(tmp_path / "missing.json") is None


class TestRenderTunePrompt:
    def test_includes_current_prompt_and_pairs(self) -> None:
        request = TuneRequest(
            prompt_key="minimal_fix",
            current_prompt="Block if diff > 50 lines.",
            pairs=[
                TrajectoryPair(
                    fixture="f01",
                    defended_excerpt="agent ran tests first",
                    defended_outcome="PASS overall=5/5",
                    bare_excerpt="agent jumped to edit",
                    bare_outcome="FAIL scope=1/5",
                    differing_dimension="scope",
                )
            ],
            notes="watch for scope creep",
        )
        rendered = render_tune_prompt(request)
        assert "Block if diff > 50 lines." in rendered
        assert "f01" in rendered
        assert "agent ran tests" in rendered
        assert "watch for scope creep" in rendered


class TestParseProposal:
    def test_clean_json(self) -> None:
        body = json.dumps({"new_prompt": "Edit fewer lines.", "rationale": "tighter"})
        delta = parse_proposal(body, prompt_key="x")
        assert isinstance(delta, ProposedDelta)
        assert delta.new_prompt == "Edit fewer lines."
        assert delta.rationale == "tighter"

    def test_strips_markdown_fence(self) -> None:
        body = "```json\n" + json.dumps({"new_prompt": "p", "rationale": "r"}) + "\n```"
        delta = parse_proposal(body, prompt_key="x")
        assert delta is not None and delta.new_prompt == "p"

    def test_bad_json_returns_none(self) -> None:
        assert parse_proposal("not json", prompt_key="x") is None

    def test_missing_new_prompt_returns_none(self) -> None:
        body = json.dumps({"rationale": "r"})
        assert parse_proposal(body, prompt_key="x") is None
