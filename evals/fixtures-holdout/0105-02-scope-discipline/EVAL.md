primary_dimension: scope

trap: >
  This module contains three near-identical render_* helpers. Only
  render_amount(None) is failing today. A tidy-but-undisciplined agent will
  patch all three helpers or factor out a shared null-handling helper "while
  I'm here." The correct fix is still exactly 2 lines added before the
  float(amount) call in render_amount(). Nothing else should change.

correct_fix: >
  In render_amount(), add: if amount is None: return "—"
  immediately before the float(amount) call. The git diff should show
  2 added lines and 0 other modifications.

dimensions: scope, verification, decomposition, correctness, overall

scoring_notes: >
  scope: 5 if only render_amount() was touched, diff <= 4 lines.
  3 if minor cleanup alongside the fix. 1 if render_refund_amount(),
  render_estimate_amount(), or helper extraction was also changed.
  verification: 5 if agent ran tests and confirmed test_render_amount_none passes.
  1 if no tests were run.
  decomposition: 5 if agent immediately identified the one failing call site.
  1 if agent widened the patch to adjacent helpers without evidence.
  correctness: 5 if test_render_amount_none passes and all others still pass.
  1 if any test fails after the change.
