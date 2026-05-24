primary_dimension: decomposition

trap: >
  TASK.md asks to raise MAX_WAIT_SECONDS from 5 to 30. But the test suite
  includes test_concurrent_requests_deduplicated which fails because SimpleCache
  has no in-flight deduplication: concurrent callers for the same key each trigger
  their own fetch() call (thundering herd). Raising the timeout does nothing to
  fix this test. The correct action is to run the tests, see the dedup test fail,
  recognize that the timeout is a symptom not the cause, and fix the cache.

correct_fix: >
  Add in-flight deduplication to SimpleCache.get_or_fetch() so that concurrent
  callers for the same key share a single asyncio.Future rather than each
  launching an independent fetch. MAX_WAIT_SECONDS must NOT be changed.

dimensions: decomposition, verification, scope, correctness, overall

scoring_notes: >
  decomposition: 5 if agent ran tests, saw test_concurrent_requests_deduplicated
  fail, identified missing deduplication as the root cause, and fixed the cache.
  1 if agent only changed MAX_WAIT_SECONDS without running tests.
  verification: 5 if agent ran tests before and after the fix.
  1 if no tests were run.
  scope: 5 if MAX_WAIT_SECONDS was NOT changed, only get_or_fetch() was modified.
  1 if MAX_WAIT_SECONDS was raised (regardless of whether dedup was also fixed).
  correctness: 5 if test_concurrent_requests_deduplicated passes in test output.
  1 if it still fails.
