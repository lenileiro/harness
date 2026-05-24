# Code Review Domain

The code-review domain is designed for repository risk review rather than implementation.

Constraints:
- it should stay read-only
- it should focus on correctness, regressions, unsafe assumptions, and missing tests
- it should return structured findings so CI and GitHub integrations can consume them reliably
