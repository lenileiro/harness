from __future__ import annotations

import json

from harness.core import parse_docs_audit_report, parse_research_memo, parse_review_report


def test_parse_review_report_parses_valid_json() -> None:
    report = parse_review_report(
        json.dumps(
            {
                "summary": "One real risk found.",
                "findings": [
                    {
                        "severity": "high",
                        "file": "src/app.py",
                        "line": 12,
                        "issue": "Side effect moved before guard.",
                        "rationale": "Can execute on invalid requests.",
                        "suggested_fix": "Move the call after validation.",
                    }
                ],
            }
        )
    )
    assert report is not None
    assert report.summary == "One real risk found."
    assert report.findings[0].file == "src/app.py"
    assert report.findings[0].line == 12


def test_parse_review_report_returns_none_for_invalid_json() -> None:
    assert parse_review_report("not json") is None


def test_parse_review_report_recovers_json_from_noisy_output() -> None:
    text = """
    leading logs
    {
      "summary": "One real risk found.",
      "findings": [
        {
          "severity": "high",
          "file": "src/app.py",
          "line": 12,
          "issue": "Side effect moved before guard.",
          "rationale": "Can execute on invalid requests."
        }
      ]
    }
    trailing runtime noise
    """
    report = parse_review_report(text)
    assert report is not None
    assert report.findings[0].file == "src/app.py"


def test_parse_research_memo_parses_valid_json() -> None:
    memo = parse_research_memo(
        json.dumps(
            {
                "summary": "SQLite is simpler for single-user local tools.",
                "findings": ["SQLite has lower operational overhead."],
                "open_questions": ["How much write concurrency is expected?"],
                "sources": [
                    {
                        "title": "SQLite docs",
                        "url": "https://sqlite.org",
                        "excerpt": "Small, fast, reliable.",
                    }
                ],
            }
        )
    )
    assert memo is not None
    assert memo.summary.startswith("SQLite is simpler")
    assert memo.sources[0].url == "https://sqlite.org"


def test_parse_research_memo_returns_none_for_invalid_json() -> None:
    assert parse_research_memo("not json") is None


def test_parse_research_memo_recovers_json_from_tool_logs() -> None:
    text = """
    → read_file(path='docs/persistence.md')
    ✓ read_file
    defense ledger:
      tools: read_file x1
    {
      "summary": "SQLite is the default persistence layer.",
      "findings": [
        "SQLite gives zero-setup local storage.",
        "SQLite has limits on concurrent writes."
      ],
      "open_questions": ["When should the system switch to Postgres?"],
      "sources": [
        {
          "title": "Persistence",
          "url": "docs/persistence.md",
          "excerpt": "zero-setup local storage with concurrent write tradeoffs"
        }
      ]
    }
    """
    memo = parse_research_memo(text)
    assert memo is not None
    assert memo.sources[0].url == "docs/persistence.md"


def test_parse_docs_audit_report_parses_valid_json() -> None:
    report = parse_docs_audit_report(
        json.dumps(
            {
                "summary": "Docs are mostly current but missing setup detail.",
                "findings": [
                    {
                        "severity": "medium",
                        "path": "README.md",
                        "issue": "No plugin setup example.",
                        "rationale": "New users cannot discover the extension flow.",
                        "suggested_update": "Add a quick plugin example.",
                    }
                ],
                "missing_topics": ["plugin setup"],
            }
        )
    )
    assert report is not None
    assert report.summary.startswith("Docs are mostly current")
    assert report.findings[0].path == "README.md"
    assert report.missing_topics == ["plugin setup"]


def test_parse_docs_audit_report_returns_none_for_invalid_json() -> None:
    assert parse_docs_audit_report("not json") is None


def test_parse_docs_audit_report_recovers_json_from_tool_logs() -> None:
    text = """
    → read_file(path='README.md')
    ✓ read_file
    defense ledger:
      tools: read_file x1
    {
      "summary": "Plugin docs are incomplete.",
      "findings": [
        {
          "severity": "medium",
          "path": "README.md",
          "issue": "Plugin setup is missing.",
          "rationale": "Users cannot discover the extension flow."
        }
      ],
      "missing_topics": ["plugin setup"]
    }
    """
    report = parse_docs_audit_report(text)
    assert report is not None
    assert report.findings[0].path == "README.md"
