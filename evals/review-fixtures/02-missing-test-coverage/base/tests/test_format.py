from src.format import normalize_email


def test_normalize_email_lowercases_and_trims() -> None:
    assert normalize_email("  A@EXAMPLE.COM ") == "a@example.com"
