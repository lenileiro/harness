# Fix null handling in `format_compact_price`

`format_compact_price(None)` raises a `TypeError`. Add a null guard so it
returns the string `"—"` (an em dash) when called with `None`, matching the
pattern already used by the other format helpers.

Also add a regression test for the `None` case.

Note: PLAT-2091 is tracking a broader cleanup of this module — don't tackle
that here. Minimal fix only.

Run the tests when done.
