"""Eval runner for the research domain."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.result_schemas import ResearchMemo, parse_research_memo


@dataclass(slots=True)
class ResearchSourceExpectation:
    title_substring: str = ""
    url_substring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchFixtureMeta:
    name: str
    path: Path
    task_text: str
    workspace_dir: Path
    required_findings: list[str] = field(default_factory=list)
    required_sources: list[ResearchSourceExpectation] = field(default_factory=list)
    summary_contains: str = ""
    min_findings: int = 1
    min_sources: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["workspace_dir"] = str(self.workspace_dir)
        return data


@dataclass(slots=True)
class ResearchEvalResult:
    fixture_name: str
    passed: bool
    findings_count: int
    source_count: int
    matched_findings: int
    matched_sources: int
    missing_expectations: list[str] = field(default_factory=list)
    summary: str = ""
    artifact_dir: Path | None = None
    raw_output: str = ""
    memo: dict[str, Any] | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_dir"] = str(self.artifact_dir) if self.artifact_dir else None
        return data


@dataclass(slots=True)
class ResearchEvalReport:
    run_id: str
    provider: str
    model: str
    artifact_root: Path
    results: list[ResearchEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "artifact_root": str(self.artifact_root),
            "results": [result.to_dict() for result in self.results],
        }


def find_research_fixtures_root(evals_root: Path | None = None) -> Path:
    root = evals_root or Path.cwd().resolve() / "evals"
    return root / "research-fixtures"


def _parse_source_expectation(item: dict[str, Any]) -> ResearchSourceExpectation:
    return ResearchSourceExpectation(
        title_substring=str(item.get("title_substring") or "").strip(),
        url_substring=str(item.get("url_substring") or "").strip(),
    )


def discover_research_fixtures(evals_root: Path | None = None) -> list[ResearchFixtureMeta]:
    fixtures_root = find_research_fixtures_root(evals_root)
    if not fixtures_root.exists():
        return []
    fixtures: list[ResearchFixtureMeta] = []
    for entry in sorted(fixtures_root.iterdir()):
        if not entry.is_dir():
            continue
        task_path = entry / "TASK.md"
        workspace_dir = entry / "workspace"
        expected_path = entry / "expected.json"
        if not (task_path.exists() and workspace_dir.is_dir() and expected_path.exists()):
            continue
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        required_sources = payload.get("required_sources") or []
        fixtures.append(
            ResearchFixtureMeta(
                name=entry.name,
                path=entry,
                task_text=task_path.read_text(encoding="utf-8").strip(),
                workspace_dir=workspace_dir,
                required_findings=[
                    str(item).strip()
                    for item in (payload.get("required_findings") or [])
                    if str(item).strip()
                ],
                required_sources=[
                    _parse_source_expectation(item)
                    for item in required_sources
                    if isinstance(item, dict)
                ],
                summary_contains=str(payload.get("summary_contains") or "").strip(),
                min_findings=int(payload.get("min_findings", 1) or 1),
                min_sources=int(payload.get("min_sources", 1) or 1),
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


def _research_cmd(
    *,
    provider: str,
    model: str,
    work: Path,
    topic: str,
    harness_bin: str | None,
    max_output_tokens: int | None,
    max_steps: int,
) -> list[str]:
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    cmd = [
        harness_cmd,
        "research",
        topic,
        "--cwd",
        str(work),
        "--provider",
        provider,
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        "--json",
        "--yes",
        "--in-memory",
    ]
    if max_output_tokens is not None:
        cmd += ["--max-output-tokens", str(max_output_tokens)]
    return cmd


def _source_matches(expectation: ResearchSourceExpectation, source: dict[str, Any]) -> bool:
    title = str(source.get("title") or "")
    url = str(source.get("url") or "")
    if expectation.title_substring and expectation.title_substring.lower() not in title.lower():
        return False
    return not (expectation.url_substring and expectation.url_substring.lower() not in url.lower())


_TOKEN_NORMALIZATIONS = {
    "concurrency": "concurrent",
    "writes": "write",
    "write": "write",
}


def _normalize_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        normalized_token = _TOKEN_NORMALIZATIONS.get(token) or token
        tokens.add(normalized_token)
    normalized: set[str] = set()
    for token in tokens:
        if token.endswith("s") and len(token) > 4:
            normalized.add(token[:-1])
        normalized.add(token)
    return normalized


def _finding_matches(expectation: str, finding: str) -> bool:
    expected_lower = expectation.lower()
    finding_lower = finding.lower()
    if expected_lower in finding_lower:
        return True
    expected_tokens = _normalize_tokens(expectation)
    finding_tokens = _normalize_tokens(finding)
    return bool(expected_tokens) and expected_tokens.issubset(finding_tokens)


def evaluate_research_memo(
    fixture: ResearchFixtureMeta,
    memo: ResearchMemo | None,
) -> tuple[bool, int, int, list[str]]:
    if memo is None:
        return False, 0, 0, ["research output was not valid JSON"]

    matched_findings = 0
    missing: list[str] = []
    finding_texts = list(memo.findings)
    for expectation in fixture.required_findings:
        if any(_finding_matches(expectation, finding) for finding in finding_texts):
            matched_findings += 1
        else:
            missing.append(f"finding~{expectation}")

    matched_sources = 0
    source_rows = [asdict(source) for source in memo.sources]
    for expectation in fixture.required_sources:
        if any(_source_matches(expectation, source) for source in source_rows):
            matched_sources += 1
        else:
            if expectation.url_substring:
                missing.append(f"source_url~{expectation.url_substring}")
            elif expectation.title_substring:
                missing.append(f"source_title~{expectation.title_substring}")
            else:
                missing.append("source~*")

    summary_ok = True
    if fixture.summary_contains:
        summary_ok = fixture.summary_contains.lower() in (memo.summary or "").lower()
        if not summary_ok:
            missing.append(f"summary~{fixture.summary_contains}")

    findings_ok = len(memo.findings) >= fixture.min_findings
    if not findings_ok:
        missing.append(f"min_findings>={fixture.min_findings}")

    sources_ok = len(memo.sources) >= fixture.min_sources
    if not sources_ok:
        missing.append(f"min_sources>={fixture.min_sources}")

    passed = (
        matched_findings == len(fixture.required_findings)
        and matched_sources == len(fixture.required_sources)
        and summary_ok
        and findings_ok
        and sources_ok
    )
    return passed, matched_findings, matched_sources, missing


def run_research_fixture(
    fixture: ResearchFixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    max_output_tokens: int | None = None,
    artifact_dir: Path | None = None,
    timeout: int = 180,
    max_steps: int = 40,
) -> ResearchEvalResult:
    with tempfile.TemporaryDirectory(prefix="harness_research_eval_") as tmp_str:
        work = Path(tmp_str) / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        _copy_tree(fixture.workspace_dir, work)

        cmd = _research_cmd(
            provider=provider,
            model=model,
            work=work,
            topic=fixture.task_text,
            harness_bin=harness_bin,
            max_output_tokens=max_output_tokens,
            max_steps=max_steps,
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

        memo = parse_research_memo(raw_output)
        passed, matched_findings, matched_sources, missing = evaluate_research_memo(
            fixture,
            memo,
        )

        result = ResearchEvalResult(
            fixture_name=fixture.name,
            passed=passed,
            findings_count=len(memo.findings) if memo is not None else 0,
            source_count=len(memo.sources) if memo is not None else 0,
            matched_findings=matched_findings,
            matched_sources=matched_sources,
            missing_expectations=missing,
            summary=memo.summary if memo is not None else "",
            artifact_dir=artifact_dir,
            raw_output=raw_output,
            memo=memo.to_dict() if memo is not None else None,
            duration_seconds=duration,
        )

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "research_output.txt").write_text(raw_output, encoding="utf-8")
            (artifact_dir / "result.json").write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
        return result


__all__ = [
    "ResearchEvalReport",
    "ResearchEvalResult",
    "ResearchFixtureMeta",
    "ResearchSourceExpectation",
    "discover_research_fixtures",
    "evaluate_research_memo",
    "find_research_fixtures_root",
    "parse_research_memo",
    "run_research_fixture",
]
