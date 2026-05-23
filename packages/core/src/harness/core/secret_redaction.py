"""Secret redaction for tool outputs.

Tools like ``shell``, ``read_file``, and ``fetch_url`` return content that
ends up in the next agent turn's context — and from there in the judge's
input. If an agent runs ``env`` (legitimately, e.g. checking PATH) or
``cat .env``, the raw output contains every API key the user has exported.
That output then:

  1. Goes into the model's next-turn context (sent to the LLM provider).
  2. Gets logged in the activity ledger (persisted to SQLite).
  3. Gets shipped to the judge for scoring (often a different provider).

Each of those is a separate data-exfiltration surface. This module strips
common secret patterns before any of those handoffs happen.

It's deliberately conservative: a missed redaction is a leak, but a false
positive only loses information. We bias toward higher-confidence patterns
(prefixed key formats like ``sk-…``, ``ghp_…``, ``AKIA…``) and explicit
env-var assignments. We do NOT try to detect "any 40-character base64
string" — too noisy.
"""

from __future__ import annotations

import re

# Each entry is (pattern, label). Match is replaced with [REDACTED:label].
# Patterns target the "key with format" class — formats published by the
# major providers, so false positives are rare.
_REDACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Env-var assignments: NAME=value where NAME contains KEY / TOKEN /
    # SECRET / PASSWORD. Catches `env` output, `.env` file dumps, etc.
    (
        re.compile(
            r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|"
            r"CREDENTIAL|CREDENTIALS|API_KEY))=([^\s\n;'\"]+)"
        ),
        "env-var",
    ),
    # Bearer tokens in Authorization headers
    (
        re.compile(r"\bBearer\s+([A-Za-z0-9._\-+/=]{8,})"),
        "bearer",
    ),
    # OpenAI / OpenRouter / Anthropic API keys (prefixed formats)
    (
        re.compile(r"\bsk-(?:ant-)?(?:api[0-9]{2}-)?[A-Za-z0-9_\-]{20,}"),
        "openai-key",
    ),
    (
        re.compile(r"\bsk-or-v[0-9]+-[A-Za-z0-9]{20,}"),
        "openrouter-key",
    ),
    (
        re.compile(r"\bsk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{20,}"),
        "anthropic-key",
    ),
    # GitHub personal-access tokens and OAuth tokens
    (
        re.compile(r"\bghp_[A-Za-z0-9]{36,}"),
        "github-pat",
    ),
    (
        re.compile(r"\bgho_[A-Za-z0-9]{36,}"),
        "github-oauth",
    ),
    (
        re.compile(r"\bghs_[A-Za-z0-9]{36,}"),
        "github-server",
    ),
    # AWS access key IDs (the SECRET_ACCESS_KEY is harder — covered by the
    # env-var pattern above when it's `AWS_SECRET_ACCESS_KEY=...`).
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws-access-key-id",
    ),
    # Slack tokens
    (
        re.compile(r"\bxox[abpros]-[A-Za-z0-9\-]{10,}"),
        "slack-token",
    ),
    # Google API keys
    (
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
        "google-api-key",
    ),
    # JWT tokens (three base64url segments separated by dots) — high false
    # positive risk in general text, so we require a "key/token-like"
    # qualifier nearby. Skipping bare JWT pattern; the env-var rule
    # catches the common JWT-in-env case.
)


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Strip known secret patterns from `text`.

    Returns ``(redacted_text, labels)`` where ``labels`` is the list of
    pattern labels that fired (e.g. ``["env-var", "openai-key"]``). When
    no patterns match, returns the original text unchanged and an empty
    list — callers can use the labels list as a "did anything redact"
    signal without re-scanning.

    A match is replaced with ``[REDACTED:label]`` (or, for env-var, with
    ``NAME=[REDACTED:env-var]`` so the variable name stays visible).
    """
    if not text:
        return text, []
    fired: list[str] = []
    result = text
    for pattern, label in _REDACTORS:
        if label == "env-var":
            # Preserve the var name; redact only the value.
            new_result, n = pattern.subn(r"\1=[REDACTED:env-var]", result)
        else:
            new_result, n = pattern.subn(f"[REDACTED:{label}]", result)
        if n > 0:
            fired.append(label)
            result = new_result
    return result, fired


def has_secrets(text: str) -> bool:
    """Cheap probe: returns True if any redactor would fire on `text`."""
    if not text:
        return False
    return any(pattern.search(text) for pattern, _ in _REDACTORS)


__all__ = ["has_secrets", "redact_secrets"]
