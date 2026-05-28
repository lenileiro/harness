from harness.adapters.openai import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
