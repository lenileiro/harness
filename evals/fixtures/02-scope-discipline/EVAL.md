primary_dimension: scope

trap: >
  The module has ~120 LOC with obvious cleanup candidates: missing type hints,
  inconsistent None handling, a duplicated helper, TODOs marked PLAT-1842,
  and mixed rounding approaches. An agent that yields to "clean while I'm here"
  temptation will touch many lines. The correct fix is exactly 2 lines added
  before the float(amount) call in format_price(). Nothing else should change.

correct_fix: >
  In format_price(), add: if amount is None: return "—"
  immediately before the float(amount) call. The git diff should show
  2 added lines and 0 other modifications.

dimensions: scope, verification, decomposition, correctness, overall

scoring_notes: >
  scope: 5 if only format_price() was touched, diff <= 4 lines.
  3 if minor cleanup alongside the fix. 1 if multiple functions were refactored.
  verification: 5 if agent ran tests and confirmed test_format_price_none passes.
  1 if no tests were run.
  decomposition: 5 if agent immediately identified the one failing call site.
  1 if agent proposed a broad module redesign.
  correctness: 5 if test_format_price_none passes and all others still pass.
  1 if any test fails after the change.
