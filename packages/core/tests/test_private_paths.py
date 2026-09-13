import os

import pytest

from harness.core.paths import read_regular_file


def test_private_read_refuses_final_symlink_without_platform_nofollow(tmp_path, monkeypatch):
    target = tmp_path / "private"
    target.write_text("secret")
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    with pytest.raises(ValueError, match="regular file"):
        read_regular_file(link, max_bytes=100)


def test_private_read_checks_file_identity_across_open_race(tmp_path, monkeypatch):
    target = tmp_path / "expected"
    target.write_text("expected")
    different = tmp_path / "different"
    different.write_text("wrong secret")
    original = os.open

    def substituted(path, flags, *args, **kwargs):
        return original(different if path == target else path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", substituted)
    with pytest.raises(ValueError, match="identity changed"):
        read_regular_file(target, max_bytes=100)
