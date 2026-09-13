"""Named local identities with isolated state, configuration and credentials."""

from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from harness.core.paths import read_regular_file

PROFILE_RESERVED_ENV = frozenset(
    {
        "HOME",
        "PATH",
        "USERPROFILE",
        "HARNESS_HOME",
        "HARNESS_PROFILE",
        "HARNESS_CONFIG",
        "HARNESS_PROFILES_ROOT",
        "CODEX_HOME",
    }
)


def read_credentials(path: Path) -> dict[str, str | None]:
    return (
        dict(
            dotenv_values(
                stream=io.StringIO(read_regular_file(path, max_bytes=1024 * 1024).decode()),
                interpolate=False,
            )
        )
        if path.exists() or path.is_symlink()
        else {}
    )


def change_credential(path: Path, name: str, value: str | None) -> None:
    """Serialize updates without following a credential-file symlink."""
    with FileLock(path.with_suffix(".lock"), timeout=30, mode=0o600):
        values = read_credentials(path)
        if value is None:
            values.pop(name, None)
        else:
            values[name] = value
        lines = []
        for key, secret in sorted(values.items()):
            if secret is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Credential file contains an invalid entry")
            quoted = secret.replace("\\", "\\\\").replace("'", "\\'")
            lines.append(f"{key}='{quoted}'\n")
        content = "".join(lines).encode()
        if len(content) > 1024 * 1024:
            raise ValueError("Credential file exceeds its size limit")
        temporary = path.with_name(f".credentials-{uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    workspace: Path
    description: str = ""


def profiles_root() -> Path:
    value = os.environ.get("HARNESS_PROFILES_ROOT")
    return Path(value).expanduser().resolve() if value else Path.home() / ".harness/profiles"


def profile_root(name: str) -> Path:
    Profile(name=name, workspace=Path.cwd())
    return profiles_root() / name


def load_profile(name: str) -> Profile:
    path = profile_root(name) / "profile.json"
    if path.parent.is_symlink():
        raise ValueError("profile directory must not be a symlink")
    profile = Profile.model_validate_json(read_regular_file(path, max_bytes=64000))
    if profile.name != name:
        raise ValueError("profile name does not match directory")
    return profile


def create_profile(name: str, *, workspace: Path | None = None, description: str = "") -> Profile:
    root = profile_root(name)
    profile = Profile(
        name=name,
        workspace=(workspace or root / "workspace").expanduser().resolve(),
        description=description,
    )
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    profile.workspace.mkdir(parents=True, exist_ok=True)
    (root / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    (root / "config.toml").write_text(
        "# Configuration for this Harness profile.\n", encoding="utf-8"
    )
    (root / "config.toml").chmod(0o600)
    return profile


def active_profile() -> str | None:
    explicit = os.environ.get("HARNESS_PROFILE")
    if explicit:
        return explicit
    selected = profiles_root() / "active.json"
    if not selected.is_file():
        return None
    value = json.loads(selected.read_text(encoding="utf-8")).get("name")
    return value if isinstance(value, str) and value else None


@contextmanager
def activate_profile(name: str) -> Iterator[Profile]:
    profile = load_profile(name)
    root = profile_root(name).resolve()
    if not profile.workspace.is_dir():
        raise ValueError(f"profile workspace is unavailable: {profile.workspace}")
    # Profiles explicitly own their credentials. Never silently borrow another
    # identity's provider keys from the launching terminal.
    credential_file = root / "credentials.env"
    credentials = read_credentials(credential_file)
    invalid = [
        key
        for key, value in credentials.items()
        if key in PROFILE_RESERVED_ENV
        or value is None
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
    ]
    if invalid:
        raise ValueError("profile credentials contain an invalid variable")
    credential_keys = {
        key
        for key in os.environ
        if re.search(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:_|$)", key)
    }
    sdk_defaults = {
        "AWS_SHARED_CREDENTIALS_FILE": str(root / "aws/credentials"),
        "AWS_CONFIG_FILE": str(root / "aws/config"),
        "AWS_PROFILE": "default",
        "AWS_DEFAULT_PROFILE": "default",
        "AWS_EC2_METADATA_DISABLED": "true",
        "CLOUDSDK_CONFIG": str(root / "gcloud"),
        "GOOGLE_APPLICATION_CREDENTIALS": str(root / "gcloud/application_default_credentials.json"),
        "AZURE_CONFIG_DIR": str(root / "azure"),
        "MODAL_CONFIG_PATH": str(root / "modal.toml"),
        "HF_HOME": str(root / "huggingface"),
    }
    managed = {
        *sdk_defaults,
        *credential_keys,
        *credentials,
        "HARNESS_HOME",
        "HARNESS_PROFILE",
        "HARNESS_CONFIG",
        "CODEX_HOME",
    }
    saved = {key: os.environ.get(key) for key in managed}
    previous_cwd = Path.cwd()
    try:
        for key in credential_keys:
            os.environ.pop(key, None)
        os.environ.update(sdk_defaults)
        os.environ.update({key: value for key, value in credentials.items() if value is not None})
        os.environ.update(
            HARNESS_HOME=str(root), HARNESS_PROFILE=name, HARNESS_CONFIG=str(root / "config.toml")
        )
        # The native Codex adapter reads its own auth store. Bind that store to
        # this identity too; the launching user's default account is not reused.
        os.environ["CODEX_HOME"] = str(root / "codex")
        os.chdir(profile.workspace)
        yield profile
    finally:
        os.chdir(previous_cwd)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
