from __future__ import annotations

from harness.core.slug import slugify


def test_slugify_produces_lowercase_ascii_slugs() -> None:
    assert slugify("Crème Brûlée") == "creme-brulee"
    assert slugify("Hello___World!!!Again") == "hello-world-again"
    assert slugify("--- spaced / value ---") == "spaced-value"
    assert slugify("   ") == "untitled"
    assert slugify("🔥🔥🔥") == "untitled"


def test_slugify_accepts_custom_fallback() -> None:
    assert slugify("🔥", fallback="item") == "item"
