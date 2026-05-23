"""Cross-framework test-output signal extraction.

Helpers for pulling structured signal out of arbitrary test-runner stdout —
failing test names, identifier vocabulary in those names. Used by the runtime
(failing-tests header in repair directives) and by verifiers (diagnosis
alignment).

Patterns cover pytest, go test, cargo test, jest/mocha, rspec, exunit. The
harness core stays language-neutral; adding a runner = adding a pattern.
"""

from __future__ import annotations

import re

_TEST_FAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # pytest:           FAILED tests/test_x.py::test_y - AssertionError
    # pytest (ERROR):   ERROR tests/test_x.py::test_y
    re.compile(r"\b(?:FAILED|ERROR)\s+([\w./:\[\]\-]+::[\w.\-\[\]]+)"),
    # go test:          --- FAIL: TestSomething (0.01s)
    re.compile(r"--- FAIL:\s+([A-Za-z_][\w/]*)"),
    # cargo test:       test foo::bar ... FAILED
    re.compile(r"\btest\s+([\w:]+)\s+\.{3,}\s*FAILED\b"),
    # jest / mocha:     ✗ test name (some runners output U+00D7 or U+2718)
    re.compile(r"(?:✗|×|✘)\s+([^\n]{3,80})"),  # noqa: RUF001 — U+00D7 is real test-runner output
    # rspec / junit:    Failure: TestName#method_name
    re.compile(r"Failure:\s+([A-Za-z_][\w.#:\-]+)"),
    # elixir exunit:    1) test name (ModuleName)
    re.compile(r"^\s+\d+\)\s+(test\s+[^\n]{3,80})", re.MULTILINE),
)


def extract_failing_test_names(text: str, limit: int = 6) -> list[str]:
    """Return up to `limit` unique failing test identifiers from any test runner."""
    if not text:
        return []
    seen: set[str] = set()
    found: list[str] = []
    for pattern in _TEST_FAIL_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            found.append(name)
            if len(found) >= limit:
                return found
    return found


# Common short tokens that carry no diagnostic signal — drop from the
# tokenized vocabulary so overlap comparisons aren't dominated by noise.
_TEST_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        "test",
        "tests",
        "spec",
        "specs",
        "should",
        "when",
        "with",
        "and",
        "or",
        "the",
        "for",
        "in",
        "to",
        "of",
        "is",
        "be",
        "a",
        "an",
        "it",
        "self",
        "cls",
        "this",
        "result",
        "case",
        "module",
        "class",
    }
)

_IDENTIFIER_SPLIT_RE = re.compile(r"[^A-Za-z]+|(?<=[a-z])(?=[A-Z])")


def tokenize_identifier(name: str, min_len: int = 3) -> set[str]:
    """Break an identifier into lowercase keyword tokens.

    Splits on non-letters AND on camelCase boundaries. Drops short tokens
    and the test-stopword set. Used to compute keyword overlap between
    failing test names and code identifiers.

    Examples:
        test_concurrent_requests_deduplicated
            → {'concurrent', 'requests', 'deduplicated'}
        TestHandlerConcurrentRequests
            → {'handler', 'concurrent', 'requests'}
        Cache#fetch_when_present
            → {'cache', 'fetch', 'when', 'present'}  (then 'when' is dropped as a stopword)
    """
    tokens = _IDENTIFIER_SPLIT_RE.split(name)
    return {
        t.lower() for t in tokens if len(t) >= min_len and t.lower() not in _TEST_NAME_STOPWORDS
    }


def keywords_for_test_names(names: list[str], min_len: int = 4) -> set[str]:
    """Union the keyword tokens across all failing test names."""
    result: set[str] = set()
    for name in names:
        result |= tokenize_identifier(name, min_len=min_len)
    return result


def text_overlap(text: str, keywords: set[str]) -> set[str]:
    """Return the subset of `keywords` whose stem appears in `text`.

    Substring match in both directions, lowercase. Handles morphological
    variation (deduplicate / deduplicated / deduplication) without a stemmer:
    a 4+ char prefix of any keyword is enough to count as a match.
    """
    if not keywords:
        return set()
    lower = text.lower()
    matches: set[str] = set()
    for kw in keywords:
        if len(kw) < 4:
            continue
        # 5-char prefix is short enough to match dedup-* variants, long enough
        # to avoid pure-stopword false positives like 'the' matching 'theme'.
        stem = kw[: max(5, len(kw) - 2)]
        if stem in lower:
            matches.add(kw)
    return matches


__all__ = [
    "extract_failing_test_names",
    "keywords_for_test_names",
    "text_overlap",
    "tokenize_identifier",
]
