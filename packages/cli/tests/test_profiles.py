from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli.__main__ import app
from harness.cli.config import default_config_path
from harness.cli.profiles import activate_profile, create_profile, profile_root
from harness.core.paths import user_home
from harness.core.skills import default_skill_paths
from harness.storage.sqlite import default_db_path


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)


def test_credentials_and_paths_are_isolated_and_restored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-secret")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "inherited-slack")
    first = create_profile("first")
    second = create_profile("second")
    (profile_root("first") / "credentials.env").write_text("OPENAI_API_KEY='first-secret'\n")
    old_cwd, old_environment = Path.cwd(), dict(os.environ)
    with activate_profile("first"):
        assert Path.cwd() == first.workspace
        assert os.environ["OPENAI_API_KEY"] == "first-secret"
        assert "SLACK_BOT_TOKEN" not in os.environ
        assert default_config_path() == user_home() / "config.toml"
        assert default_db_path().is_relative_to(user_home())
        assert default_skill_paths(Path.cwd())[-1] == user_home() / "skills"
        assert Path(os.environ["CODEX_HOME"]).is_relative_to(user_home())
        with activate_profile("second"):
            assert Path.cwd() == second.workspace
            assert "OPENAI_API_KEY" not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "first-secret"
    assert Path.cwd() == old_cwd
    assert dict(os.environ) == old_environment


def test_cli_profile_setup_and_auth_do_not_leak_or_modify_other_profile():
    runner = CliRunner()
    for name in ("first", "second"):
        result = runner.invoke(app, ["profiles", "create", name])
        assert result.exit_code == 0, result.output
    cwd = Path.cwd()
    result = runner.invoke(
        app,
        ["--profile", "first", "setup", "--provider", "ollama", "--model", "example", "--force"],
    )
    assert result.exit_code == 0, result.output
    assert Path.cwd() == cwd
    assert 'model = "example"' in (profile_root("first") / "config.toml").read_text()
    assert "example" not in (profile_root("second") / "config.toml").read_text()
    result = runner.invoke(
        app, ["auth", "set", "EXAMPLE_API_KEY", "--profile", "first"], input="private-value\n"
    )
    assert result.exit_code == 0, result.output
    assert "private-value" not in result.output
    result = runner.invoke(app, ["auth", "status", "--profile", "first"])
    assert "EXAMPLE_API_KEY" in result.output and "private-value" not in result.output


def test_invalid_profile_credentials_fail_before_changing_identity():
    create_profile("broken")
    (profile_root("broken") / "credentials.env").write_text("HOME='/unexpected'\n")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="invalid variable"), activate_profile("broken"):
        pytest.fail("profile should not activate")
    assert dict(os.environ) == before


def test_profile_names_cannot_escape_storage():
    with pytest.raises(ValueError):
        create_profile("../escape")


def test_profiles_clear_compound_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "borrowed-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "borrowed-secret")
    create_profile("cloud")
    with activate_profile("cloud"):
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert os.environ["AWS_ACCESS_KEY_ID"] == "borrowed-id"


def test_profile_sdk_stores_are_private_and_explicit_credentials_can_override(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AWS_PROFILE", "personal")
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "personal-modal.toml"))
    create_profile("cloud")
    with activate_profile("cloud"):
        for key in (
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CONFIG_DIR",
            "MODAL_CONFIG_PATH",
            "HF_HOME",
        ):
            assert Path(os.environ[key]).is_relative_to(user_home())
        assert os.environ["AWS_PROFILE"] == "default"
        assert os.environ["AWS_EC2_METADATA_DISABLED"] == "true"
    assert os.environ["AWS_PROFILE"] == "personal"
    from harness.cli.profiles import change_credential

    explicit = str(tmp_path / "selected-cloud.json")
    change_credential(
        profile_root("cloud") / "credentials.env", "GOOGLE_APPLICATION_CREDENTIALS", explicit
    )
    with activate_profile("cloud"):
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == explicit


def test_credential_updates_preserve_literals_and_refuse_symlinks(tmp_path):
    from harness.cli.profiles import change_credential, read_credentials

    path = tmp_path / "credentials.env"
    secret = "apostrophe' backslash\\ double\" newline\n${LITERAL_TOKEN}"
    change_credential(path, "FIRST_TOKEN", secret)
    change_credential(path, "SECOND_TOKEN", "second")
    assert read_credentials(path) == {"FIRST_TOKEN": secret, "SECOND_TOKEN": "second"}
    change_credential(path, "SECOND_TOKEN", None)
    assert read_credentials(path) == {"FIRST_TOKEN": secret}
    link = tmp_path / "linked.env"
    link.symlink_to(path)
    with pytest.raises((OSError, ValueError)):
        change_credential(link, "FIRST_TOKEN", "replacement")
    assert read_credentials(path)["FIRST_TOKEN"] == secret


@pytest.mark.parametrize("name", ["CODEX_HOME", "USERPROFILE", "HARNESS_PROFILES_ROOT"])
def test_auth_cannot_store_reserved_runtime_settings(name):
    create_profile("safe")
    result = CliRunner().invoke(app, ["auth", "set", name, "--profile", "safe", "--value", "bad"])
    assert result.exit_code != 0
    assert not (profile_root("safe") / "credentials.env").exists()
