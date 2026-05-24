from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    severity: str
    file: str
    line: int | None
    issue: str
    rationale: str
    suggested_fix: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewReport:
    summary: str
    findings: list[ReviewFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchSource:
    title: str
    url: str
    excerpt: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchMemo:
    summary: str
    findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    sources: list[ResearchSource] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocsAuditFinding:
    severity: str
    path: str | None
    issue: str
    rationale: str
    suggested_update: str | None = None


@dataclass(frozen=True, slots=True)
class DocsAuditReport:
    summary: str
    findings: list[DocsAuditFinding] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_fenced_json(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        parts = body.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
    return body


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    body = _normalize_wrapped_json_strings(_strip_fenced_json(text))
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(body):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(body[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    return candidates


def _normalize_wrapped_json_strings(text: str) -> str:
    parts: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                parts.append(char)
                escaped = False
                continue
            if char == "\\":
                parts.append(char)
                escaped = True
                continue
            if char == '"':
                parts.append(char)
                in_string = False
                continue
            if char in "\r\n":
                if parts and parts[-1] != " ":
                    parts.append(" ")
                continue
            parts.append(char)
            continue
        parts.append(char)
        if char == '"':
            in_string = True
            escaped = False
    return "".join(parts)


def parse_review_report(text: str) -> ReviewReport | None:
    for payload in reversed(_json_object_candidates(text)):
        summary = str(payload.get("summary") or "").strip()
        findings_payload = payload.get("findings") or []
        if not isinstance(findings_payload, list):
            continue
        findings: list[ReviewFinding] = []
        for item in findings_payload:
            if not isinstance(item, dict):
                continue
            issue = str(item.get("issue") or "").strip()
            rationale = str(item.get("rationale") or "").strip()
            file = str(item.get("file") or "").strip()
            if not issue or not rationale or not file:
                continue
            line_value = item.get("line")
            line = int(line_value) if isinstance(line_value, int) else None
            findings.append(
                ReviewFinding(
                    severity=str(item.get("severity") or "medium").strip().lower(),
                    file=file,
                    line=line,
                    issue=issue,
                    rationale=rationale,
                    suggested_fix=(
                        str(item.get("suggested_fix")).strip()
                        if item.get("suggested_fix") is not None
                        else None
                    ),
                )
            )
        if summary or findings:
            return ReviewReport(summary=summary, findings=findings)
    return None


def parse_research_memo(text: str) -> ResearchMemo | None:
    for payload in reversed(_json_object_candidates(text)):
        summary = str(payload.get("summary") or "").strip()
        findings_payload = payload.get("findings") or []
        questions_payload = payload.get("open_questions") or []
        sources_payload = payload.get("sources") or []
        if not isinstance(findings_payload, list) or not isinstance(questions_payload, list):
            continue
        if not isinstance(sources_payload, list):
            continue
        findings = [str(item).strip() for item in findings_payload if str(item).strip()]
        open_questions = [str(item).strip() for item in questions_payload if str(item).strip()]
        sources: list[ResearchSource] = []
        for item in sources_payload:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            excerpt = str(item.get("excerpt")).strip() if item.get("excerpt") is not None else None
            sources.append(ResearchSource(title=title, url=url, excerpt=excerpt))
        if summary or findings or sources or open_questions:
            return ResearchMemo(
                summary=summary,
                findings=findings,
                open_questions=open_questions,
                sources=sources,
            )
    return None


def parse_docs_audit_report(text: str) -> DocsAuditReport | None:
    for payload in reversed(_json_object_candidates(text)):
        summary = str(payload.get("summary") or "").strip()
        findings_payload = payload.get("findings") or []
        topics_payload = payload.get("missing_topics") or []
        if not isinstance(findings_payload, list) or not isinstance(topics_payload, list):
            continue
        findings: list[DocsAuditFinding] = []
        for item in findings_payload:
            if not isinstance(item, dict):
                continue
            issue = str(item.get("issue") or "").strip()
            rationale = str(item.get("rationale") or "").strip()
            if not issue or not rationale:
                continue
            path_value = str(item.get("path") or "").strip() or None
            findings.append(
                DocsAuditFinding(
                    severity=str(item.get("severity") or "medium").strip().lower(),
                    path=path_value,
                    issue=issue,
                    rationale=rationale,
                    suggested_update=(
                        str(item.get("suggested_update")).strip()
                        if item.get("suggested_update") is not None
                        else None
                    ),
                )
            )
        missing_topics = [str(item).strip() for item in topics_payload if str(item).strip()]
        if summary or findings or missing_topics:
            return DocsAuditReport(
                summary=summary,
                findings=findings,
                missing_topics=missing_topics,
            )
    return None


__all__ = [
    "DocsAuditFinding",
    "DocsAuditReport",
    "ResearchMemo",
    "ResearchSource",
    "ReviewFinding",
    "ReviewReport",
    "parse_docs_audit_report",
    "parse_research_memo",
    "parse_review_report",
]
