"""Consistent private backups, validated restores, and checkout maintenance."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import typer

from harness.core.paths import read_regular_file, user_home

maintenance_app = typer.Typer(
    help="Back up, restore, and maintain a Harness installation.", no_args_is_help=True
)
MAX_BACKUP_BYTES = 1024 * 1024 * 1024
MAX_BACKUP_FILES = 10000


def _private_credential_path(path: Path) -> bool:
    return bool(
        {part.casefold() for part in path.parts}
        & {
            "auth",
            "codex",
            ".codex",
            ".ssh",
            "credentials.env",
            ".env",
            "credentials.json",
            "aws",
            "gcloud",
            "azure",
            "huggingface",
            "modal.toml",
            "weixin-account.json",
            "photon-account.json",
        }
    )


def create_backup(
    source: Path,
    destination: Path,
    *,
    include_credentials: bool = False,
    additional_sources: Mapping[str, Path] | None = None,
) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() and not additional_sources:
        raise ValueError("Backup source is not a directory")
    if destination.is_relative_to(source):
        raise ValueError("Write the backup outside its source directory")
    if destination.exists():
        raise ValueError("Backup destination already exists")
    entries: dict[Path, Path] = {}
    excluded = {".git", ".venv", "__pycache__", "node_modules"}

    def collect(root: Path, prefix: Path) -> None:
        if root.is_symlink():
            return
        if root.is_file():
            previous = entries.get(prefix)
            if previous is not None and previous.resolve() != root.resolve():
                raise ValueError("Backup source paths conflict")
            entries[prefix] = root
            return
        for directory, dirs, names in os.walk(root, followlinks=False):
            base = Path(directory)
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in excluded
                and not (base / name).is_symlink()
                and (
                    include_credentials
                    or not _private_credential_path(prefix / (base / name).relative_to(root))
                )
            )
            for name in sorted(names):
                path = base / name
                if not path.is_symlink() and path.is_file():
                    entries[prefix / path.relative_to(root)] = path

    collect(source, Path())
    for name, path in (additional_sources or {}).items():
        logical = PurePosixPath(name)
        if (
            logical.is_absolute()
            or ".." in logical.parts
            or "\\" in name
            or ":" in name
            or not logical.parts
        ):
            raise ValueError("Additional backup paths must be relative")
        if destination.is_relative_to(path.resolve()):
            raise ValueError("Write the backup outside all source directories")
        collect(path, Path(logical))
    files = []
    total = 0
    for relative, path in sorted(entries.items()):
        if excluded & set(relative.parts):
            continue
        if not include_credentials and _private_credential_path(relative):
            continue
        if path.name.endswith(("-wal", "-shm", ".lock")):
            continue
        if relative.as_posix() == "harness-backup-manifest.json":
            raise ValueError("Backup source contains the reserved manifest filename")
        total += path.stat().st_size
        if total > MAX_BACKUP_BYTES or len(files) >= MAX_BACKUP_FILES:
            raise ValueError("Backup exceeds 1 GiB or 10000 files")
        files.append((relative, path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "credentials_included": include_credentials, "files": []}
    with tempfile.TemporaryDirectory(prefix="harness-backup-") as temporary:
        staging = Path(temporary)
        archive = staging / "backup.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            snapshot_total = 0
            for index, (relative, path) in enumerate(files):
                raw = read_regular_file(path, max_bytes=MAX_BACKUP_BYTES)
                if raw.startswith(b"SQLite format 3\0"):
                    snapshot = staging / f"{index}.db"
                    database = sqlite3.connect(
                        f"file:{quote(str(path))}?mode=ro", uri=True, timeout=10
                    )
                    target = sqlite3.connect(snapshot)
                    try:
                        database.backup(target)
                    finally:
                        target.close()
                        database.close()
                    raw = read_regular_file(snapshot, max_bytes=MAX_BACKUP_BYTES)
                snapshot_total += len(raw)
                if snapshot_total > MAX_BACKUP_BYTES:
                    raise ValueError("Consistent database snapshots exceed 1 GiB")
                info = tarfile.TarInfo(relative.as_posix())
                info.size, info.mode = len(raw), 0o700 if os.access(path, os.X_OK) else 0o600
                bundle.addfile(info, io.BytesIO(raw))
                manifest["files"].append(
                    {
                        "path": relative.as_posix(),
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            raw = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo("harness-backup-manifest.json")
            info.size, info.mode = len(raw), 0o600
            bundle.addfile(info, io.BytesIO(raw))
        with destination.open("xb") as output, archive.open("rb") as source_file:
            destination.chmod(0o600)
            shutil.copyfileobj(source_file, output)
    return manifest


def restore_backup(archive: Path, destination: Path) -> dict:
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Restore destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".harness-restore-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "restored"
        staged.mkdir(mode=0o700)
        with tarfile.open(archive, "r:gz") as bundle:
            members = []
            expanded_size = 0
            for member in bundle:
                members.append(member)
                expanded_size += member.size
                if (
                    len(members) > MAX_BACKUP_FILES + 1
                    or expanded_size > MAX_BACKUP_BYTES + 4 * 1024 * 1024
                ):
                    raise ValueError("Backup exceeds restore limits")
            names: set[str] = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or ":" in member.name
                    or not path.parts
                    or member.name in names
                    or path.as_posix() != member.name
                ):
                    raise ValueError("Backup contains an unsafe or duplicate entry")
                names.add(member.name)
            metadata = bundle.extractfile("harness-backup-manifest.json")
            if metadata is None:
                raise ValueError("Backup manifest is missing")
            raw_manifest = metadata.read(4 * 1024 * 1024 + 1)
            if len(raw_manifest) > 4 * 1024 * 1024:
                raise ValueError("Backup manifest exceeds its size limit")
            manifest = json.loads(raw_manifest)
            if manifest.get("version") != 1:
                raise ValueError("Unsupported backup version")
            expected = {item["path"]: item for item in manifest["files"]}
            if len(expected) != len(manifest["files"]) or set(expected) != names - {
                "harness-backup-manifest.json"
            }:
                raise ValueError("Backup manifest does not match archive")
            for member in members:
                if member.name == "harness-backup-manifest.json":
                    continue
                source = bundle.extractfile(member)
                assert source is not None
                raw = source.read(member.size + 1)
                record = expected[member.name]
                if (
                    len(raw) != record["size"]
                    or hashlib.sha256(raw).hexdigest() != record["sha256"]
                ):
                    raise ValueError("Backup checksum mismatch")
                target = staged / member.name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(raw)
                target.chmod(0o700 if member.mode & 0o100 else 0o600)
        staged.rename(destination)
    return manifest


@maintenance_app.command("backup")
def backup(
    destination: Path, include_credentials: bool = typer.Option(False, "--include-credentials")
) -> None:
    """Back up this identity, taking consistent snapshots of SQLite databases."""
    try:
        from harness.cli.config import default_config_path
        from harness.storage.sqlite import default_db_path

        root = user_home()
        # The default identity historically stores config and sessions in XDG
        # directories. Profiles may also point at a workspace outside their home.
        candidates = {
            "config.toml": default_config_path(),
            "state/sessions.db": default_db_path(),
            "workspace/.harness": Path.cwd() / ".harness",
        }
        extra = {
            name: path
            for name, path in candidates.items()
            if path.exists() and not path.resolve().is_relative_to(root.resolve())
        }
        manifest = create_backup(
            root, destination, include_credentials=include_credentials, additional_sources=extra
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved {len(manifest['files'])} files to {destination}")


@maintenance_app.command("restore")
def restore(archive: Path, destination: Path = typer.Option(..., "--destination")) -> None:
    """Verify and restore into a new directory; existing identities are never overwritten."""
    try:
        manifest = restore_backup(archive, destination)
    except (ValueError, OSError, tarfile.TarError, KeyError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Restored {len(manifest['files'])} verified files to {destination}")


@maintenance_app.command("update")
def update(
    check: bool = typer.Option(False, "--check"),
    checkout: Path | None = typer.Option(None, "--checkout"),
) -> None:
    """Check or fast-forward an existing clean source checkout, then sync its locked dependencies."""
    root = (checkout or Path(__file__).resolve().parents[5]).resolve()
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        raise typer.BadParameter("Provide --checkout pointing to the Harness source repository")
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
    )
    if result.stdout.strip():
        raise typer.BadParameter("Commit or save local changes before updating the checkout")
    subprocess.run(["git", "fetch", "--prune"], cwd=root, check=True)
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD..@{upstream}"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    typer.echo(f"{result.stdout.strip()} upstream commits available")
    if not check:
        subprocess.run(["git", "merge", "--ff-only", "@{upstream}"], cwd=root, check=True)
        subprocess.run(
            ["uv", "sync", "--frozen", "--all-extras", "--all-groups"], cwd=root, check=True
        )
