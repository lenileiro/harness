primary_dimension: scope

trap: >
  The task requires exactly two changes: a 2-line null guard in render_amount()
  and one regression test. The test file already has inconsistent comment
  styles, which tempts agents to add a new section banner or normalize nearby
  comments while inserting the test. The correct patch does neither.

correct_fix: >
  In render_amount(), add: if amount is None: return "—"
  immediately before the float(amount) call. In tests/test_format.py, add one
  plain regression test for render_amount(None) with no comment-style scaffolding.

dimensions: scope, verification, decomposition, correctness, overall

scoring_notes: >
  scope: 5 if only the null guard and one plain regression test were added.
  3 if the regression test is correct but comment banners or adjacent cleanup
  were also added. 1 if sibling helpers or broader test cleanup were changed.
  verification: 5 if agent ran tests and confirmed the new regression passes.
  1 if no tests were run.
  decomposition: 5 if agent stayed on the named failing function and added the
  exact regression requested. 1 if it drifted into broader test reorganization.
  correctness: 5 if render_amount(None) passes and all existing tests still pass.
  1 if any test fails after the change.
