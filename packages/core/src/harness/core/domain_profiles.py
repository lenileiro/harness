from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.core.extensions import DomainProfileProvider


@dataclass(frozen=True, slots=True)
class DomainProfile:
    name: str
    description: str
    allowed_tools: tuple[str, ...] | None = None
    system_prompt: str | None = None
    output_schema: str | None = None


_CODE_REVIEW_PROMPT = """
You are operating in code-review mode.

Your job is to review the proposed change, not to implement it.
Do not make edits. Read the diff and any relevant files, then identify real risks:
- correctness bugs
- behavioral regressions
- missing tests
- unsafe assumptions
- trust-boundary or side-effect issues

Ignore style-only nits and low-value commentary.
Prefer a short findings list over speculative noise.

Return JSON only with this shape:
{
  "summary": "one short paragraph",
  "findings": [
    {
      "severity": "high|medium|low",
      "file": "relative/path.py",
      "line": 12,
      "issue": "what is wrong",
      "rationale": "why it matters",
      "suggested_fix": "optional concrete fix"
    }
  ]
}
""".strip()

_RESEARCH_PROMPT = """
You are operating in research mode.

Your job is to investigate the topic, gather the most decision-useful information,
and return a concise structured memo.

Rules:
- Use repository files first when the workspace is likely to contain the answer
- Do not browse the web unless the repository clearly does not contain enough evidence
- Gather only the minimum evidence needed; stop once you have 1-3 strong sources
- As soon as you have enough evidence, produce the final JSON answer immediately
- Prefer high-signal findings over exhaustive coverage
- Surface uncertainty and open questions explicitly
- Do not write code unless the user explicitly asks for it
- When citing repository files as sources, use the repo-relative file path as the source URL

Return JSON only with this shape:
{
  "summary": "one short paragraph",
  "findings": ["key finding 1", "key finding 2"],
  "open_questions": ["question 1", "question 2"],
  "sources": [
    {
      "title": "source title",
      "url": "https://example.com",
      "excerpt": "optional short excerpt"
    }
  ]
}
""".strip()

_DOCS_AUDIT_PROMPT = """
You are operating in docs-audit mode.

Your job is to inspect repository documentation and identify the highest-value
gaps, stale claims, or unclear sections.

Rules:
- Stay read-only
- Prefer concrete documentation issues over style commentary
- Call out missing topics explicitly when they matter for onboarding or safe use
- Use repo-relative paths when citing documentation files

Return JSON only with this shape:
{
  "summary": "one short paragraph",
  "findings": [
    {
      "severity": "high|medium|low",
      "path": "README.md",
      "issue": "what is wrong or missing",
      "rationale": "why it matters",
      "suggested_update": "optional concrete update suggestion"
    }
  ],
  "missing_topics": ["topic 1", "topic 2"]
}
""".strip()


_PROFILES: dict[str, DomainProfile] = {
    "coding": DomainProfile(
        name="coding",
        description="Default coding and workspace execution domain.",
    ),
    "code-review": DomainProfile(
        name="code-review",
        description="Read-only code review over repo diffs and changed files.",
        allowed_tools=("read_file", "list_dir", "glob", "fetch_url", "web_search"),
        system_prompt=_CODE_REVIEW_PROMPT,
        output_schema="review_report",
    ),
    "research": DomainProfile(
        name="research",
        description="Read-only research and synthesis across repo files and web sources.",
        allowed_tools=("read_file", "list_dir", "glob"),
        system_prompt=_RESEARCH_PROMPT,
        output_schema="research_memo",
    ),
    "docs-audit": DomainProfile(
        name="docs-audit",
        description="Read-only documentation audit over repository docs and references.",
        allowed_tools=("read_file", "list_dir", "glob", "fetch_url", "web_search"),
        system_prompt=_DOCS_AUDIT_PROMPT,
        output_schema="docs_audit_report",
    ),
}


def _merged_profiles(
    providers: list[DomainProfileProvider] | None = None,
) -> dict[str, DomainProfile]:
    profiles = dict(_PROFILES)
    for provider in providers or []:
        for profile in provider.profiles():
            profiles[profile.name] = profile
    return profiles


def get_domain_profile(
    name: str,
    *,
    providers: list[DomainProfileProvider] | None = None,
) -> DomainProfile:
    profiles = _merged_profiles(providers)
    try:
        return profiles[name]
    except KeyError as exc:
        raise KeyError(f"unknown domain profile {name!r}") from exc


def domain_profile_names(*, providers: list[DomainProfileProvider] | None = None) -> list[str]:
    return sorted(_merged_profiles(providers))


__all__ = ["DomainProfile", "domain_profile_names", "get_domain_profile"]
