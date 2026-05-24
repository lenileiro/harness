"""Eval runner for the code-review domain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.result_schemas import ReviewReport, parse_review_report


@dataclass(slots=True)
class ReviewFindingExpectation:
    file: str
    severity: str | None = None
    line: int | None = None
    issue_substring: str = ""
    rationale_substring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewFixtureMeta:
    name: str
    path: Path
    task_text: str
    base_dir: Path
    head_dir: Path
    required_findings: list[ReviewFindingExpectation] = field(default_factory=list)
    summary_contains: str = ""
    min_findings: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["base_dir"] = str(self.base_dir)
        data["head_dir"] = str(self.head_dir)
        return data


@dataclass(slots=True)
class ReviewEvalResult:
    fixture_name: str
    passed: bool
    findings_count: int
    matched_expectations: int
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
class ReviewEvalReport:
    run_id: str
    provider: str
    model: str
    artifact_root: Path
    results: list[ReviewEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "artifact_root": str(self.artifact_root),
            "results": [result.to_dict() for result in self.results],
        }


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "eval",
            "GIT_AUTHOR_EMAIL": "eval@harness",
            "GIT_COMMITTER_NAME": "eval",
            "GIT_COMMITTER_EMAIL": "eval@harness",
        }
    )
    return env


def find_review_fixtures_root(evals_root: Path | None = None) -> Path:
    root = evals_root or Path.cwd().resolve() / "evals"
    return root / "review-fixtures"


def _parse_expectation(item: dict[str, Any]) -> ReviewFindingExpectation:
    return ReviewFindingExpectation(
        file=str(item.get("file") or "").strip(),
        severity=(str(item["severity"]).strip().lower() if item.get("severity") else None),
        line=int(item["line"]) if isinstance(item.get("line"), int) else None,
        issue_substring=str(item.get("issue_substring") or "").strip(),
        rationale_substring=str(item.get("rationale_substring") or "").strip(),
    )


def discover_review_fixtures(evals_root: Path | None = None) -> list[ReviewFixtureMeta]:
    fixtures_root = find_review_fixtures_root(evals_root)
    if not fixtures_root.exists():
        return []
    fixtures: list[ReviewFixtureMeta] = []
    for entry in sorted(fixtures_root.iterdir()):
        if not entry.is_dir():
            continue
        task_path = entry / "TASK.md"
        base_dir = entry / "base"
        head_dir = entry / "head"
        expected_path = entry / "expected.json"
        if not (
            task_path.exists()
            and base_dir.is_dir()
            and head_dir.is_dir()
            and expected_path.exists()
        ):
            continue
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        required = payload.get("required_findings") or []
        fixtures.append(
            ReviewFixtureMeta(
                name=entry.name,
                path=entry,
                task_text=task_path.read_text(encoding="utf-8"),
                base_dir=base_dir,
                head_dir=head_dir,
                required_findings=[
                    _parse_expectation(item) for item in required if isinstance(item, dict)
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


def _review_cmd(
    *,
    provider: str,
    model: str,
    work: Path,
    harness_bin: str | None,
    max_output_tokens: int | None,
) -> list[str]:
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    cmd = [
        harness_cmd,
        "review",
        "--cwd",
        str(work),
        "--base",
        "HEAD~1",
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


def _matches(expectation: ReviewFindingExpectation, finding: dict[str, Any]) -> bool:
    if expectation.file and str(finding.get("file") or "").strip() != expectation.file:
        return False
    if (
        expectation.severity
        and str(finding.get("severity") or "").strip().lower() != expectation.severity
    ):
        return False
    if expectation.line is not None and finding.get("line") != expectation.line:
        return False
    issue = str(finding.get("issue") or "")
    rationale = str(finding.get("rationale") or "")
    if expectation.issue_substring and expectation.issue_substring.lower() not in issue.lower():
        return False
    if expectation.rationale_substring:
        return expectation.rationale_substring.lower() in rationale.lower()
    return True


def evaluate_review_report(
    fixture: ReviewFixtureMeta,
    report: ReviewReport | None,
) -> tuple[bool, int, list[str]]:
    if report is None:
        return False, 0, ["review output was not valid JSON"]
    findings = [
        finding.__dict__ if hasattr(finding, "__dict__") else asdict(finding)
        for finding in report.findings
    ]
    matched = 0
    missing: list[str] = []
    for expectation in fixture.required_findings:
        if any(_matches(expectation, finding) for finding in findings):
            matched += 1
        else:
            missing.append(
                f"{expectation.file}:{expectation.severity or '*'}:{expectation.issue_substring or expectation.rationale_substring or '*'}"
            )
    summary_ok = True
    if fixture.summary_contains:
        summary_ok = fixture.summary_contains.lower() in (report.summary or "").lower()
        if not summary_ok:
            missing.append(f"summary~{fixture.summary_contains}")
    findings_ok = len(report.findings) >= fixture.min_findings
    if not findings_ok:
        missing.append(f"min_findings>={fixture.min_findings}")
    passed = matched == len(fixture.required_findings) and summary_ok and findings_ok
    return passed, matched, missing


def run_review_fixture(
    fixture: ReviewFixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    max_output_tokens: int | None = None,
    artifact_dir: Path | None = None,
    timeout: int = 180,
) -> ReviewEvalResult:
    with tempfile.TemporaryDirectory(prefix="harness_review_eval_") as tmp_str:
        work = Path(tmp_str) / "repo"
        work.mkdir(parents=True, exist_ok=True)
        _copy_tree(fixture.base_dir, work)
        git_env = _git_env()
        subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        subprocess.run(["git", "add", "-A"], cwd=work, env=git_env, capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", "base", "--no-verify"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        _copy_tree(fixture.head_dir, work)
        subprocess.run(["git", "add", "-A"], cwd=work, env=git_env, capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", "head", "--no-verify"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        cmd = _review_cmd(
            provider=provider,
            model=model,
            work=work,
            harness_bin=harness_bin,
            max_output_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        duration = time.perf_counter() - started
        output = result.stdout + result.stderr
        report = parse_review_report(result.stdout)
        passed, matched, missing = evaluate_review_report(fixture, report)
        eval_result = ReviewEvalResult(
            fixture_name=fixture.name,
            passed=(result.returncode == 0 and passed),
            findings_count=len(report.findings) if report is not None else 0,
            matched_expectations=matched,
            missing_expectations=missing,
            summary=report.summary if report is not None else "",
            artifact_dir=artifact_dir,
            raw_output=output,
            report=(report.to_dict() if report is not None else None),
            duration_seconds=duration,
        )
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "review_output.txt").write_text(output, encoding="utf-8")
            (artifact_dir / "result.json").write_text(
                json.dumps(eval_result.to_dict(), indent=2),
                encoding="utf-8",
            )
        return eval_result


__all__ = [
    "ReviewEvalReport",
    "ReviewEvalResult",
    "ReviewFindingExpectation",
    "ReviewFixtureMeta",
    "discover_review_fixtures",
    "evaluate_review_report",
    "find_review_fixtures_root",
    "run_review_fixture",
]
