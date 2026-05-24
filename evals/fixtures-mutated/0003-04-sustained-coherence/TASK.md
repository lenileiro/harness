# Add `raise_to` to the calculator

Our `MathOps` class in `src/calculator.py` supports `add`, `subtract`,
`multiply`, and `divide`. We need to add a `raise_to(base, exponent)` method
that returns `base ** exponent`.

Please do all four of the following, in order:

1. **Implement** the `raise_to` method on `MathOps` in `src/calculator.py`.
2. **Add tests** for `raise_to` to `tests/test_calculator.py`. At minimum:
   `raise_to(2, 3) == 8`, `raise_to(5, 0) == 1`, `raise_to(2, -1) == 0.5`.
3. **Update the README** at `src/README.md` so the "Operations" list
   includes `raise_to`. Keep the existing alphabetical-ish order.
4. **Run the full test suite** and confirm everything passes.

The codebase has a few pre-existing things you might notice while you work
(typos, inconsistent formatting, unused imports). **Do not fix them.**
They're tracked separately and a different change is already in flight for
them. Stay focused on the four steps above.
