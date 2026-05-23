"""Tests for L3 — action canonicalization."""

from __future__ import annotations

from harness.core import canonicalize_tool_name


class TestExactAndCaseInsensitive:
    def test_exact_match_returns_unchanged(self) -> None:
        result = canonicalize_tool_name("read_file", known=["read_file", "shell"])
        assert not result.changed
        assert result.confidence == 1.0
        assert result.reason == "exact_match"

    def test_case_insensitive_match(self) -> None:
        """LLMs often capitalize tool names — `Read` should resolve to `read_file`."""
        result = canonicalize_tool_name("Read_File", known=["read_file"])
        assert result.canonical == "read_file"
        assert result.reason == "case_insensitive_match"


class TestAliasTable:
    def test_bash_resolves_to_shell(self) -> None:
        result = canonicalize_tool_name("bash", known=["shell"])
        assert result.canonical == "shell"
        assert result.reason.startswith("alias:bash->shell")

    def test_read_resolves_to_read_file(self) -> None:
        result = canonicalize_tool_name("read", known=["read_file"])
        assert result.canonical == "read_file"
        assert result.changed

    def test_alias_only_used_when_target_registered(self) -> None:
        """If the alias's target isn't registered, fall through (no invention)."""
        result = canonicalize_tool_name("bash", known=["python"])
        # Falls through to fuzzy and then no_match.
        assert result.canonical != "shell"


class TestFuzzyMatch:
    def test_typo_finds_close_match(self) -> None:
        result = canonicalize_tool_name("read_fil", known=["read_file", "shell"])
        assert result.canonical == "read_file"
        assert result.reason.startswith("fuzzy_match")

    def test_no_close_match_falls_through(self) -> None:
        result = canonicalize_tool_name("frobnicate", known=["read_file", "shell"])
        assert not result.changed
        assert result.reason == "no_match"
        assert result.confidence == 0.0


class TestCustomAliases:
    def test_overriding_alias_table(self) -> None:
        custom = {"do_thing": "shell"}
        result = canonicalize_tool_name("do_thing", known=["shell"], aliases=custom)
        assert result.canonical == "shell"
