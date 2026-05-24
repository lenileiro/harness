# Fix null handling in render_amount

`render_amount(None)` raises a `TypeError`. Add a null guard so it returns
the string `"—"` (an em dash) when called with `None`, matching the pattern
already used by the other format functions.

Important: the `render_refund_amount()` and `render_estimate_amount()` helpers
have their own follow-up ticket for None-handling cleanup. Do not widen this
patch to those helpers or any other module cleanup. Minimal fix only.

Run the tests when done.
