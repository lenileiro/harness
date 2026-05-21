# QA Report — harness — 2026-05-21

**Type:** Full quality gate (Python CLI/library — no browser)
**Branch:** main
**Duration:** ~3 min
**Health Score:** Baseline 94 → Final 97

---

## Summary

| Gate | Result |
|---|---|
| Tests | 371/371 passed |
| Lint (ruff check) | Clean |
| Format (ruff format) | Clean (1 file fixed) |
| Types (pyright) | 0 errors, 0 warnings |
| Coverage | 91.6% overall |
| CLI smoke | All 9 commands, all error paths |

---

## Issues Found

### ISSUE-001 — Stale README status line
**Severity:** Medium  
**Category:** Content  
**Status:** Fixed

README said "scaffolding phase. Nothing is runnable yet." The project is fully functional with 371 tests, a working CLI, and live Ollama integration verified today.

**Fix:** Updated status line to "functional. Install with `uv sync`, then run `harness --help`."

---

### ISSUE-002 — Formatting violation in test_task_lifecycle.py
**Severity:** Low  
**Category:** Code quality  
**Status:** Fixed

`ruff format --check` flagged the new lifecycle test file (list literals not reformatted to multi-line style). Auto-formatted with `ruff format`.

---

### ISSUE-003 — cli/approval.py coverage at 39%
**Severity:** Low  
**Category:** Test coverage  
**Status:** Deferred

`RichApprovalHandler` (lines 29–72) has no test coverage. The interactive TTY path (Rich `Prompt.ask`) and the non-TTY guard both go untested. These require a simulated terminal or monkeypatching `sys.stdin.isatty` + `Prompt.ask`. Not a bug — deferred to a future test pass.

---

### ISSUE-004 — core/telemetry.py coverage at 79%
**Severity:** Low  
**Category:** Test coverage  
**Status:** Deferred

OpenTelemetry span context manager paths and the `NoopTracer` fallback (lines 89–99) are uncovered. Low risk since they're instrumentation-only — uncovered paths are silent no-ops by design.

---

## CLI Smoke Results

All commands exercised; all error paths return exit code 1 with a clear message:

| Command | Result |
|---|---|
| `harness version` | ✓ |
| `harness providers list` | ✓ (ollama ready, openrouter key missing) |
| `harness tools list` | ✓ |
| `harness sessions list` | ✓ |
| `harness sessions show <nonexistent>` | ✓ exit 1 + "not found" |
| `harness tasks list` | ✓ |
| `harness tasks show T-999` | ✓ exit 1 + "not found" |
| `harness approvals list` | ✓ IDs no longer truncated |
| `harness approvals grant <bad-id>` | ✓ exit 1 + "not found" |
| `harness evidence list --session` | ✓ |

---

## Coverage Summary

| File | Coverage | Notes |
|---|---|---|
| core/activity.py | 100% | |
| core/budget.py | 100% | |
| core/schemas.py | 100% | |
| core/tools.py | 100% | |
| tasks/* | 100% | |
| storage-memory | 99% | |
| storage-sqlite | 97% | |
| adapter-ollama | 93% | |
| core/runtime.py | 89% | Error/cancel branches |
| cli/__main__.py | 89% | Config file paths, verbose mode |
| **cli/approval.py** | **39%** | Interactive TTY handler — deferred |
| **core/telemetry.py** | **79%** | OTEL spans — deferred |
| **TOTAL** | **91.6%** | |

---

## Health Score Breakdown

| Category | Score | Weight | Contribution |
|---|---|---|---|
| Tests | 100 | 20% | 20.0 |
| Types | 100 | 15% | 15.0 |
| Lint/Format | 100 | 10% | 10.0 |
| Coverage | 88 | 15% | 13.2 |
| CLI UX | 98 | 15% | 14.7 |
| Content/Docs | 95 | 10% | 9.5 |
| Error handling | 100 | 15% | 15.0 |
| **TOTAL** | **97** | | |

**QA found 4 issues, fixed 2, health score 94 → 97.**
