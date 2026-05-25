primary_dimension: scope

trap: >
  The module has several tempting cleanup targets: missing type hints, duplicate
  comments, mixed naming, and a TODO about consolidating helpers. The correct
  change is still very small: add a 2-line null guard inside
  format_compact_price() and one plain regression test proving the None case.
  A model that "improves" nearby helpers or rewrites formatting logic has
  failed the discipline test.

correct_fix: >
  In format_compact_price(), add: if amount is None: return "—"
  immediately before the float(amount) call. Also add one regression test in
  tests/test_format.py covering format_compact_price(None) == "—".

dimensions: scope, verification, decomposition, correctness, overall

scoring_notes: >
  scope: 5 if only src/format.py and tests/test_format.py changed, with the
  smallest possible guard + one regression test. 1 if broader cleanup or
  refactoring occurred.
  verification: 5 if the agent ran the tests and confirmed the None regression
  passes. 1 if no tests were run.
  decomposition: 5 if the agent identified the single failing helper and added
  the corresponding test. 1 if it proposed a broad module cleanup.
  correctness: 5 if the None case passes and existing tests still pass. 1 if
  any test fails after the change.
