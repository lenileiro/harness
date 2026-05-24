# Fix null handling in render_amount

`render_amount(None)` raises a `TypeError`. Add a null guard so it returns
the string `"—"` (an em dash) when called with `None`, matching the pattern
already used by the other format functions.

Also add one regression test covering `render_amount(None) == "—"`.

Important: the mixed comment styles already present in `tests/test_format.py`
and the broader render_* helper cleanup are tracked separately. Do not tidy
comments, add section banners, or widen the patch beyond the null guard plus
the one regression test.

Run the tests when done.
