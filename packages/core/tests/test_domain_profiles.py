from __future__ import annotations

import pytest

from harness.core import DomainProfile, domain_profile_names, get_domain_profile


def test_domain_profile_names_include_code_review() -> None:
    assert domain_profile_names() == [
        "code-review",
        "coding",
        "docs-audit",
        "mission-planning",
        "research",
    ]


def test_code_review_profile_is_read_only() -> None:
    profile = get_domain_profile("code-review")
    assert profile.output_schema == "review_report"
    assert profile.allowed_tools == (
        "read_file",
        "list_dir",
        "glob",
        "fetch_url",
        "web_search",
    )
    assert "Return JSON only" in (profile.system_prompt or "")


def test_research_profile_is_read_only_and_structured() -> None:
    profile = get_domain_profile("research")
    assert profile.output_schema == "research_memo"
    assert profile.allowed_tools == ("read_file", "list_dir", "glob")
    assert "Return JSON only" in (profile.system_prompt or "")


def test_docs_audit_profile_is_read_only_and_structured() -> None:
    profile = get_domain_profile("docs-audit")
    assert profile.output_schema == "docs_audit_report"
    assert profile.allowed_tools == (
        "read_file",
        "list_dir",
        "glob",
        "fetch_url",
        "web_search",
    )
    assert "Return JSON only" in (profile.system_prompt or "")


def test_mission_planning_profile_is_structured() -> None:
    profile = get_domain_profile("mission-planning")
    assert profile.output_schema == "mission_plan_draft"
    assert profile.allowed_tools == ("read_file", "list_dir", "glob")
    assert "Return JSON only" in (profile.system_prompt or "")


def test_unknown_domain_profile_raises() -> None:
    with pytest.raises(KeyError):
        get_domain_profile("nope")


def test_provider_profiles_extend_and_override_names() -> None:
    class DemoProvider:
        def profiles(self) -> list[DomainProfile]:
            return [
                DomainProfile(
                    name="code-review",
                    description="override",
                    allowed_tools=("read_file",),
                ),
                DomainProfile(name="docs-review", description="docs-only"),
            ]

    assert domain_profile_names(providers=[DemoProvider()]) == [
        "code-review",
        "coding",
        "docs-audit",
        "docs-review",
        "mission-planning",
        "research",
    ]
    profile = get_domain_profile("code-review", providers=[DemoProvider()])
    assert profile.allowed_tools == ("read_file",)
