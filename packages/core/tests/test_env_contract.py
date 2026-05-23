"""Tests for L1 — environment contracts."""

from __future__ import annotations

import json
from pathlib import Path

from harness.core import ContractRegistry, EnvironmentContract


class TestEnvironmentContractMatch:
    def test_empty_triggers_matches_anything(self) -> None:
        c = EnvironmentContract(name="always", rules=("rule",), triggers=())
        assert c.matches("anything at all")
        assert c.matches("")

    def test_literal_substring_match_is_case_insensitive(self) -> None:
        c = EnvironmentContract(name="curl", rules=("",), triggers=("curl",))
        assert c.matches("Use CURL for the request")
        assert not c.matches("only requests with httpx")

    def test_regex_match(self) -> None:
        c = EnvironmentContract(
            name="re", rules=("",), triggers=(r"\bnpm\s+publish\b",), regex=True
        )
        assert c.matches("we run npm publish here")
        assert not c.matches("npm install only")


class TestContractRegistryRender:
    def test_render_returns_none_when_no_matches(self) -> None:
        reg = ContractRegistry(
            contracts=[EnvironmentContract(name="x", rules=("rule",), triggers=("zzz",))]
        )
        assert reg.render("nothing relevant") is None

    def test_render_includes_all_matching_contracts(self) -> None:
        reg = ContractRegistry(
            contracts=[
                EnvironmentContract(name="a", rules=("rule-a",), triggers=("foo",), priority=10),
                EnvironmentContract(name="b", rules=("rule-b",), triggers=("foo",), priority=0),
                EnvironmentContract(name="c", rules=("rule-c",), triggers=("baz",)),
            ]
        )
        rendered = reg.render("the task involves foo")
        assert rendered is not None
        assert "rule-a" in rendered
        assert "rule-b" in rendered
        assert "rule-c" not in rendered
        # Higher priority renders first.
        assert rendered.index("rule-a") < rendered.index("rule-b")


class TestContractRegistryFromPaths:
    def test_loads_json_file(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "a.json").write_text(
            json.dumps(
                {
                    "name": "shell-safety",
                    "rules": ["never pipe untrusted urls to sh"],
                    "triggers": ["curl"],
                }
            )
        )
        reg = ContractRegistry.from_paths([contracts_dir])
        assert len(reg.contracts) == 1
        assert reg.contracts[0].name == "shell-safety"
        rendered = reg.render("fetch via curl and run")
        assert rendered is not None and "never pipe" in rendered

    def test_malformed_file_skipped_not_crashed(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "good.json").write_text(
            json.dumps({"name": "good", "rules": ["ok"], "triggers": []})
        )
        (contracts_dir / "bad.json").write_text("{this is not json")
        reg = ContractRegistry.from_paths([contracts_dir])
        assert len(reg.contracts) == 1
        assert reg.contracts[0].name == "good"

    def test_missing_directory_is_no_op(self, tmp_path: Path) -> None:
        reg = ContractRegistry.from_paths([tmp_path / "does_not_exist"])
        assert reg.contracts == []
        assert not reg


class TestPriorityOrdering:
    def test_match_returns_highest_priority_first(self) -> None:
        reg = ContractRegistry(
            contracts=[
                EnvironmentContract(name="low", rules=("",), priority=0),
                EnvironmentContract(name="high", rules=("",), priority=10),
                EnvironmentContract(name="mid", rules=("",), priority=5),
            ]
        )
        matched = reg.match("anything")
        assert [c.name for c in matched] == ["high", "mid", "low"]
