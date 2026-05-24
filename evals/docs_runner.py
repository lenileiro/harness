"""Eval runner for the docs-audit domain."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.result_schemas import DocsAuditReport, parse_docs_audit_report


@dataclass(slots=True)
class DocsFindingExpectation:
    path: str | None = None
    severity: str | None = None
    issue_substring: str = ""
    rationale_substring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocsFixtureMeta:
    name: str
    path: Path
    task_text: str
    workspace_dir: Path
    required_findings: list[DocsFindingExpectation] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    summary_contains: str = ""
    min_findings: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["workspace_dir"] = str(self.workspace_dir)
        return data


@dataclass(slots=True)
class DocsEvalResult:
    fixture_name: str
    passed: bool
    findings_count: int
    matched_expectations: int
    matched_topics: int
    missing_expectations: list[str] = field(default_factory=list)
    summary: str = ""
    artifact_dir: Path | None = None
    raw_output: str = ""
    report: dict[str, Any] | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_dir"] = str(self.artifact_dir) if self.artifact_dir else None
        return data


@dataclass(slots=True)
class DocsEvalReport:
    run_id: str
    provider: str
    model: str
    artifact_root: Path
    results: list[DocsEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "artifact_root": str(self.artifact_root),
            "results": [result.to_dict() for result in self.results],
        }


def find_docs_fixtures_root(evals_root: Path | None = None) -> Path:
    root = evals_root or Path.cwd().resolve() / "evals"
    return root / "docs-fixtures"


def _parse_expectation(item: dict[str, Any]) -> DocsFindingExpectation:
    path = str(item.get("path") or "").strip() or None
    severity = str(item.get("severity") or "").strip().lower() or None
    return DocsFindingExpectation(
        path=path,
        severity=severity,
        issue_substring=str(item.get("issue_substring") or "").strip(),
        rationale_substring=str(item.get("rationale_substring") or "").strip(),
    )


def discover_docs_fixtures(evals_root: Path | None = None) -> list[DocsFixtureMeta]:
    fixtures_root = find_docs_fixtures_root(evals_root)
    if not fixtures_root.exists():
        return []
    fixtures: list[DocsFixtureMeta] = []
    for entry in sorted(fixtures_root.iterdir()):
        if not entry.is_dir():
            continue
        task_path = entry / "TASK.md"
        workspace_dir = entry / "workspace"
        expected_path = entry / "expected.json"
        if not (task_path.exists() and workspace_dir.is_dir() and expected_path.exists()):
            continue
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        fixtures.append(
            DocsFixtureMeta(
                name=entry.name,
                path=entry,
                task_text=task_path.read_text(encoding="utf-8").strip(),
                workspace_dir=workspace_dir,
                required_findings=[
                    _parse_expectation(item)
                    for item in (payload.get("required_findings") or [])
                    if isinstance(item, dict)
                ],
                missing_topics=[
                    str(item).strip()
                    for item in (payload.get("missing_topics") or [])
                    if str(item).strip()
                ],
                summary_contains=str(payload.get("summary_contains") or "").strip(),
                min_findings=int(payload.get("min_findings", 1) or 1),
            )
        )
    return fixtures


def _copy_tree(src: Path, dest: Path) -> None:
    for child in src.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _docs_cmd(
    *,
    provider: str,
    model: str,
    work: Path,
    focus: str,
    harness_bin: str | None,
    max_output_tokens: int | None,
) -> list[str]:
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    cmd = [
        harness_cmd,
        "docs-audit",
        focus,
        "--cwd",
        str(work),
        "--provider",
        provider,
        "--model",
        model,
        "--json",
        "--yes",
        "--in-memory",
    ]
    if max_output_tokens is not None:
        cmd += ["--max-output-tokens", str(max_output_tokens)]
    return cmd


def _matches(expectation: DocsFindingExpectation, finding: dict[str, Any]) -> bool:
    if expectation.path is not None and str(finding.get("path") or "").strip() != expectation.path:
        return False
    if (
        expectation.severity
        and str(finding.get("severity") or "").strip().lower() != expectation.severity
    ):
        return False
    issue = str(finding.get("issue") or "")
    rationale = str(finding.get("rationale") or "")
    if expectation.issue_substring and expectation.issue_substring.lower() not in issue.lower():
        return False
    if expectation.rationale_substring:
        return expectation.rationale_substring.lower() in rationale.lower()
    return True


def evaluate_docs_report(
    fixture: DocsFixtureMeta,
    report: DocsAuditReport | None,
) -> tuple[bool, int, int, list[str]]:
    if report is None:
        return False, 0, 0, ["docs output was not valid JSON"]
    findings = [asdict(finding) for finding in report.findings]
    matched = 0
    missing: list[str] = []
    for expectation in fixture.required_findings:
        if any(_matches(expectation, finding) for finding in findings):
            matched += 1
        else:
            missing.append(
                f"{expectation.path or '*'}:{expectation.severity or '*'}:{expectation.issue_substring or expectation.rationale_substring or '*'}"
            )

    matched_topics = 0
    for topic in fixture.missing_topics:
        if any(topic.lower() == value.lower() for value in report.missing_topics):
            matched_topics += 1
        else:
            missing.append(f"missing_topic~{topic}")

    summary_ok = True
    if fixture.summary_contains:
        summary_ok = fixture.summary_contains.lower() in (report.summary or "").lower()
        if not summary_ok:
            missing.append(f"summary~{fixture.summary_contains}")
    findings_ok = len(report.findings) >= fixture.min_findings
    if not findings_ok:
        missing.append(f"min_findings>={fixture.min_findings}")
    passed = (
        matched == len(fixture.required_findings)
        and matched_topics == len(fixture.missing_topics)
        and summary_ok
        and findings_ok
    )
    return passed, matched, matched_topics, missing


def run_docs_fixture(
    fixture: DocsFixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    max_output_tokens: int | None = None,
    artifact_dir: Path | None = None,
    timeout: int = 180,
) -> DocsEvalResult:
    with tempfile.TemporaryDirectory(prefix="harness_docs_eval_") as tmp_str:
        work = Path(tmp_str) / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        _copy_tree(fixture.workspace_dir, work)

        cmd = _docs_cmd(
            provider=provider,
            model=model,
            work=work,
            focus=fixture.task_text,
            harness_bin=harness_bin,
            max_output_tokens=max_output_tokens,
        )

        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - started
        raw_output = proc.stdout.strip()
        if proc.returncode != 0 and proc.stderr.strip():
            raw_output = (raw_output + "\n" + proc.stderr.strip()).strip()

        report = parse_docs_audit_report(raw_output)
        passed, matched, matched_topics, missing = evaluate_docs_report(fixture, report)

        result = DocsEvalResult(
            fixture_name=fixture.name,
            passed=passed,
            findings_count=len(report.findings) if report is not None else 0,
            matched_expectations=matched,
            matched_topics=matched_topics,
            missing_expectations=missing,
            summary=report.summary if report is not None else "",
            artifact_dir=artifact_dir,
            raw_output=raw_output,
            report=report.to_dict() if report is not None else None,
            duration_seconds=duration,
        )

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "docs_output.txt").write_text(raw_output, encoding="utf-8")
            (artifact_dir / "result.json").write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
        return result


__all__ = [
    "DocsEvalReport",
    "DocsEvalResult",
    "DocsFindingExpectation",
    "DocsFixtureMeta",
    "discover_docs_fixtures",
    "evaluate_docs_report",
    "find_docs_fixtures_root",
    "parse_docs_audit_report",
    "run_docs_fixture",
]
