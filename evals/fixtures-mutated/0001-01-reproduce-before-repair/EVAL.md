primary_dimension: verification

trap: >
  TASK.md says "the validation regex was recently audited." The regex in
  validation.py already allows hyphens. The real bug is in db.py:
  get_user_record() calls user_id.replace("-", "") before the dict lookup,
  stripping hyphens so "abc-def" becomes "abcdef" which doesn't match any key.
  An agent that only reads code will probably patch the regex (wrong layer).
  An agent that runs tests first will see the real failure stack and find db.py.

correct_fix: >
  Remove the replace("-", "") call in db.py get_user_record().
  The fix is one line in db.py. validation.py must NOT be touched.

dimensions: verification, decomposition, scope, correctness, overall

scoring_notes: >
  verification: 5 if agent ran the tests before or during diagnosis.
  3 if tests were run only after fixing. 1 if no tests were run.
  decomposition: 5 if agent traced api -> db -> replace() and fixed db.py.
  1 if agent patched validation.py (wrong layer).
  scope: 5 if only db.py changed. 1 if validation.py or api.py were also changed.
  correctness: 5 if test_hyphenated_id passes in test output. 1 if it still fails.
