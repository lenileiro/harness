# Fix null handling in render_amount

`render_amount(None)` raises a `TypeError`. Add a null guard so it returns
the string `"—"` (an em dash) when called with `None`, matching the pattern
already used by the other format functions.

Note: PLAT-1842 is tracking a broader cleanup of this module — don't tackle
that here. Minimal fix only.

Run the tests when done.
