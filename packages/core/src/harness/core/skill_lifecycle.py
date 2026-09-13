"""Versioned skill packages and evidence-backed, approval-reviewed evolution.

Archives are data: no checkout hooks, installers, or scripts are executed. Run
evidence records outcomes, not a claim that a model's conclusion was correct.
"""

from __future__ import annotations

import base64
import difflib
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from filelock import FileLock

from harness.core.activity import ActivityEvent
from harness.core.schemas import ApprovalDecision, Session, ToolCall, ToolResult
from harness.core.skills import (
    MAX_PACKAGE_BYTES,
    MAX_SKILL_BYTES,
    Skill,
    SkillError,
    SkillLibrary,
    validate_skill_name,
)
from harness.core.tools import Tool

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _no_links(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SkillError("skill lifecycle paths must not traverse symlinks")


def _read_bytes(path: Path, limit: int) -> bytes:
    _no_links(path)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SkillError("skill package entries must be regular files")
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise SkillError("skill package exceeds its byte limit")
    return value


def _tree(directory: Path) -> dict[str, dict[str, Any]]:
    _no_links(directory)
    Skill.load(directory)
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        _no_links(path)
        if path.is_dir():
            continue
        data = _read_bytes(path, MAX_PACKAGE_BYTES - total)
        total += len(data)
        files[path.relative_to(directory).as_posix()] = {
            "data": base64.b64encode(data).decode("ascii"),
            "executable": bool(path.stat().st_mode & 0o111),
        }
        if len(files) > 1000:
            raise SkillError("skill package exceeds 1000 files")
    return files


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise SkillError("archive paths must be relative and cannot traverse parents")
    return path


def git_archive_url(repository: str, commit: str) -> tuple[str, str]:
    """Only public HTTPS GitHub/GitLab archives pinned to a full commit ID."""
    if not _COMMIT.fullmatch(commit):
        raise SkillError("commit must be a full lowercase 40-character Git commit ID")
    parsed = urlsplit(repository)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in {"github.com", "gitlab.com"}
        or parsed.query
        or parsed.fragment
    ):
        raise SkillError("repository must be a public HTTPS github.com or gitlab.com URL")
    path = parsed.path.strip("/").removesuffix(".git")
    parts = path.split("/")
    if len(parts) < 2 or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in {".", ".."} for p in parts
    ):
        raise SkillError("invalid repository path")
    canonical = f"https://{parsed.netloc}/{path}"
    if parsed.netloc == "github.com":
        if len(parts) != 2:
            raise SkillError("GitHub repository URL must contain owner/repository only")
        return canonical, f"https://api.github.com/repos/{path}/tarball/{commit}"
    return canonical, (
        f"https://gitlab.com/api/v4/projects/{quote(path, safe='')}/repository/archive.tar.gz?sha={commit}"
    )


class _ArchiveRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.netloc not in {"codeload.github.com", "gitlab.com"}:
            raise SkillError("archive redirect left the approved HTTPS archive hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "Harness-Skill-Installer", "Accept": "application/vnd.github+json"},
    )
    try:
        with build_opener(_ArchiveRedirect()).open(request, timeout=30) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise SkillError("could not download the pinned public skill archive") from exc
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SkillError("compressed archive exceeds 32 MiB")
    return data


def _extract_package(archive: bytes, subpath: str, destination: Path) -> None:
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise SkillError("compressed archive exceeds 32 MiB")
    selected = _relative(subpath)
    if not selected.parts:
        selected = PurePosixPath(".")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive)) as stream:
            expanded = stream.read(MAX_EXPANDED_BYTES + 1)
        if len(expanded) > MAX_EXPANDED_BYTES:
            raise SkillError("expanded archive exceeds 64 MiB")
        seen: set[str] = set()
        total = 0
        archive_root: str | None = None
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as tar:
            for index, entry in enumerate(tar):
                if index >= 10000:
                    raise SkillError("archive exceeds 10000 entries")
                relative = _relative(entry.name)
                if not relative.parts:
                    continue
                if archive_root is None:
                    archive_root = relative.parts[0]
                if relative.parts[0] != archive_root:
                    raise SkillError("repository archive must have one root directory")
                if not entry.isfile() and not entry.isdir():
                    raise SkillError("archive links and special files are forbidden")
                package_path = PurePosixPath(*relative.parts[1:])
                if not package_path.is_relative_to(selected):
                    continue
                target_relative = package_path.relative_to(selected)
                if entry.isdir():
                    continue
                key = str(target_relative)
                if not target_relative.parts or key in seen:
                    raise SkillError("duplicate or invalid archive file")
                seen.add(key)
                total += entry.size
                if total > MAX_PACKAGE_BYTES or len(seen) > 1000:
                    raise SkillError("selected skill exceeds 16 MiB or 1000 files")
                target = destination.joinpath(*target_relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                member = tar.extractfile(entry)
                if member is None:
                    raise SkillError("unreadable archive entry")
                with member:
                    data = member.read(entry.size + 1)
                if len(data) != entry.size:
                    raise SkillError("truncated archive entry")
                target.write_bytes(data)
                target.chmod(0o755 if entry.mode & 0o111 else 0o644)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise SkillError("invalid skill archive") from exc
    Skill.load(destination)


class SkillLifecycle:
    """One installed root, with locked SQLite history outside the visible library."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.history = self.root.parent / "skill-history" / self.root.name
        _no_links(self.root)
        _no_links(self.history)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.history / "history.db"
        _no_links(self.database)
        _no_links(self.history / "history.lock")
        self.lock = FileLock(str(self.history / "history.lock"), timeout=10)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS revisions (
                  name TEXT, digest TEXT, files TEXT NOT NULL, provenance TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(name,digest));
                CREATE TABLE IF NOT EXISTS heads (name TEXT PRIMARY KEY, digest TEXT);
                CREATE TABLE IF NOT EXISTS journal (name TEXT PRIMARY KEY, old TEXT, new TEXT);
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS changes (
                  id INTEGER PRIMARY KEY, name TEXT NOT NULL, old TEXT, new TEXT,
                  provenance TEXT NOT NULL, state TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            """)
        self.database.chmod(0o600)

    @contextmanager
    def _db(self):
        _no_links(self.database)
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _actual(self, name: str) -> dict[str, dict[str, Any]] | None:
        directory = self.root / validate_skill_name(name)
        _no_links(directory)
        return _tree(directory) if directory.exists() else None

    def _revision(
        self, db: sqlite3.Connection, name: str, revision: str
    ) -> dict[str, dict[str, Any]]:
        row = db.execute(
            "SELECT files FROM revisions WHERE name=? AND digest=?", (name, revision)
        ).fetchone()
        if row is None:
            raise SkillError("unknown skill revision")
        files = json.loads(row["files"])
        if _digest(files) != revision:
            raise SkillError("skill revision integrity check failed")
        return files

    def _publish(self, name: str, files: dict[str, dict[str, Any]] | None) -> None:
        destination = self.root / name
        _no_links(destination)
        with tempfile.TemporaryDirectory(prefix=".skill-change-", dir=self.root) as temp:
            stage = Path(temp) / name
            if files is not None:
                stage.mkdir()
                for relative, record in files.items():
                    path = stage.joinpath(*_relative(relative).parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(base64.b64decode(record["data"], validate=True))
                    path.chmod(0o755 if record["executable"] else 0o644)
                _tree(stage)
            backup = Path(temp) / ".previous"
            if destination.exists():
                destination.rename(backup)
            try:
                if files is not None:
                    stage.rename(destination)
            except BaseException:
                if backup.exists():
                    backup.rename(destination)
                raise

    def _recover(self, db: sqlite3.Connection) -> None:
        for row in db.execute("SELECT name,old,new FROM journal").fetchall():
            name, old, new = row["name"], row["old"], row["new"]
            actual = self._actual(name)
            digest = _digest(actual) if actual is not None else None
            if digest not in {old, new, None}:
                raise SkillError(
                    f"interrupted change to {name} conflicts with current files; preserve those files before recovery"
                )
            if digest != new:
                self._publish(name, self._revision(db, name, new) if new else None)
            db.execute("INSERT OR REPLACE INTO heads VALUES (?,?)", (name, new))
            db.execute(
                "UPDATE changes SET state='applied' WHERE name=? AND state='pending'", (name,)
            )
            db.execute("DELETE FROM journal WHERE name=?", (name,))
        db.commit()

    def _commit(
        self,
        name: str,
        files: dict[str, dict[str, Any]] | None,
        provenance: dict[str, Any],
        *,
        expected: str | None,
    ) -> dict[str, Any]:
        validate_skill_name(name)
        with self.lock, self._db() as db:
            self._recover(db)
            current = self._actual(name)
            actual = _digest(current) if current is not None else None
            if actual != expected:
                raise SkillError(
                    "skill changed since inspection; inspect the current revision before retrying"
                )
            if current is not None:
                db.execute(
                    "INSERT OR IGNORE INTO revisions(name,digest,files,provenance) VALUES (?,?,?,?)",
                    (
                        name,
                        actual,
                        _json(current),
                        _json({"kind": "existing", "source": "operator-installed package"}),
                    ),
                )
            revision = _digest(files) if files is not None else None
            if files is not None:
                db.execute(
                    "INSERT OR IGNORE INTO revisions(name,digest,files,provenance) VALUES (?,?,?,?)",
                    (name, revision, _json(files), _json(provenance)),
                )
            db.execute("INSERT INTO journal VALUES (?,?,?)", (name, actual, revision))
            db.execute(
                "INSERT INTO changes(name,old,new,provenance,state) VALUES (?,?,?,?,?)",
                (name, actual, revision, _json(provenance), "pending"),
            )
            db.commit()
            self._publish(name, files)
            db.execute("INSERT OR REPLACE INTO heads VALUES (?,?)", (name, revision))
            db.execute(
                "UPDATE changes SET state='applied' WHERE name=? AND state='pending'", (name,)
            )
            db.execute("DELETE FROM journal WHERE name=?", (name,))
        return self.inspect(name)

    def inspect(self, name: str) -> dict[str, Any]:
        validate_skill_name(name)
        with self.lock, self._db() as db:
            self._recover(db)
            files = self._actual(name)
            actual = _digest(files) if files is not None else None
            row = db.execute("SELECT digest FROM heads WHERE name=?", (name,)).fetchone()
            tracked = row["digest"] if row else None
            revisions = [
                dict(r)
                for r in db.execute(
                    "SELECT digest,provenance,created_at FROM revisions WHERE name=? ORDER BY rowid DESC",
                    (name,),
                )
            ]
            for revision in revisions:
                revision["provenance"] = json.loads(revision["provenance"])
            changes = [
                dict(r)
                for r in db.execute(
                    "SELECT old,new,provenance,state,created_at FROM changes WHERE name=? ORDER BY id DESC",
                    (name,),
                )
            ]
            for change in changes:
                change["provenance"] = json.loads(change["provenance"])
            return {
                "name": name,
                "path": str(self.root / name),
                "installed": files is not None,
                "revision": actual,
                "tracked_revision": tracked,
                "modified": row is not None and actual != tracked,
                "revisions": revisions,
                "changes": changes,
            }

    def install(self, source: Path) -> dict[str, Any]:
        skill = Skill.load(source)
        return self._commit(
            skill.name,
            _tree(source),
            {"kind": "local", "path": str(source.absolute())},
            expected=None,
        )

    def install_git(
        self,
        name: str,
        repository: str,
        commit: str,
        subpath: str,
        *,
        expected: str | None = None,
        fetch: Callable[[str], bytes] = download_archive,
    ) -> dict[str, Any]:
        validate_skill_name(name)
        canonical, url = git_archive_url(repository, commit)
        _relative(subpath)
        archive = fetch(url)
        with tempfile.TemporaryDirectory(prefix="harness-skill-archive-") as temp:
            directory = Path(temp).resolve() / name
            directory.mkdir()
            _extract_package(archive, subpath, directory)
            files = _tree(directory)
        return self._commit(
            name,
            files,
            {
                "kind": "git",
                "repository": canonical,
                "commit": commit,
                "path": subpath,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
            },
            expected=expected,
        )

    def update(self, name: str, source: Path, *, expected: str) -> dict[str, Any]:
        if Skill.load(source).name != name:
            raise SkillError("updated package must keep the installed skill name")
        return self._commit(
            name,
            _tree(source),
            {"kind": "local", "path": str(source.absolute())},
            expected=expected,
        )

    def remove(self, name: str, *, expected: str) -> dict[str, Any]:
        return self._commit(name, None, {"kind": "remove"}, expected=expected)

    def rollback(self, name: str, revision: str, *, expected: str | None) -> dict[str, Any]:
        with self.lock, self._db() as db:
            self._recover(db)
            files = self._revision(db, name, revision)
        return self._commit(
            name, files, {"kind": "rollback", "revision": revision}, expected=expected
        )

    def save_evidence(self, evidence: dict[str, Any]) -> str:
        identifier = _digest(evidence)
        with self.lock, self._db() as db:
            db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?)", (identifier, _json(evidence)))
        return identifier

    def evidence(self, name: str | None = None) -> list[dict[str, Any]]:
        if name is not None:
            validate_skill_name(name)
        with self._db() as db:
            records = [
                {"id": r["id"], **json.loads(r["data"])}
                for r in db.execute("SELECT id,data FROM evidence ORDER BY rowid DESC")
            ]
        return [record for record in records if record["skill"] == name]

    def propose(
        self,
        name: str,
        content: str,
        *,
        session_id: str,
        evidence_ids: list[str],
        reason: str,
        create: bool = False,
    ) -> dict[str, Any]:
        if (
            not content
            or len(content.encode()) > MAX_SKILL_BYTES
            or not reason
            or len(reason) > 2000
        ):
            raise SkillError(
                "proposal needs bounded skill content and a reason of 1-2000 characters"
            )
        with self.lock, self._db() as db:
            self._recover(db)
            files = self._actual(name)
            if files is None and not create:
                raise SkillError("cannot evolve a skill that is not installed")
            if files is not None and create:
                raise SkillError("cannot create a skill that already exists")
            base = _digest(files) if files is not None else "absent"
            old = base64.b64decode(files["SKILL.md"]["data"]).decode() if files is not None else ""
            if old == content:
                raise SkillError("proposal does not change the skill")
            if not evidence_ids or len(evidence_ids) > 20:
                raise SkillError("proposal must cite 1-20 stored run evidence IDs")
            for identifier in evidence_ids:
                row = db.execute("SELECT data FROM evidence WHERE id=?", (identifier,)).fetchone()
                if row is None or json.loads(row["data"])["skill"] not in (
                    {None, name} if create else {name}
                ):
                    raise SkillError("proposal evidence must refer to this skill's recorded runs")
            with tempfile.TemporaryDirectory(prefix="harness-skill-proposal-") as temp:
                directory = Path(temp).resolve() / name
                directory.mkdir()
                (directory / "SKILL.md").write_text(content)
                Skill.load(directory)
            diff = "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile="/dev/null" if create else f"{name}/SKILL.md (current)",
                    tofile=f"{name}/SKILL.md (proposed)",
                )
            )
            proposal = {
                "name": name,
                "base": base,
                "content": content,
                "reason": reason,
                "session_id": session_id,
                "evidence": sorted(set(evidence_ids)),
                "diff": diff,
                "create": create,
            }
            identifier = _digest(proposal)
            db.execute(
                "INSERT OR IGNORE INTO proposals VALUES (?,?)", (identifier, _json(proposal))
            )
        return {"id": identifier, **proposal}

    def proposal(self, identifier: str) -> dict[str, Any]:
        if not _SHA.fullmatch(identifier):
            raise SkillError("invalid proposal ID")
        with self._db() as db:
            row = db.execute("SELECT data FROM proposals WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise SkillError("unknown proposal")
        proposal = json.loads(row["data"])
        if _digest(proposal) != identifier:
            raise SkillError("proposal integrity check failed")
        return {"id": identifier, **proposal}

    def apply(
        self, identifier: str, *, expected: str, reviewed_diff: str, session_id: str | None = None
    ) -> dict[str, Any]:
        proposal = self.proposal(identifier)
        if proposal["base"] != expected or proposal["diff"] != reviewed_diff:
            raise SkillError("approved base and diff must exactly match the stored proposal")
        if session_id is not None and proposal["session_id"] != session_id:
            raise SkillError("proposal belongs to a different session")
        files = self._actual(proposal["name"])
        create = proposal.get("create", False)
        if create:
            if files is not None or expected != "absent":
                raise SkillError("skill appeared since proposal; review its current contents")
            files = {"SKILL.md": {"data": "", "executable": False}}
        elif files is None or _digest(files) != expected:
            raise SkillError("skill changed since proposal; create and review a new proposal")
        files["SKILL.md"]["data"] = base64.b64encode(proposal["content"].encode()).decode()
        return self._commit(
            proposal["name"],
            files,
            {
                "kind": "creation" if create else "evolution",
                "proposal_id": identifier,
                "session_id": proposal["session_id"],
                "evidence": proposal["evidence"],
            },
            expected=None if create else expected,
        )


def _local(session: Session) -> bool:
    scope = session.metadata.get("memory_scope", {})
    return isinstance(scope, dict) and scope.get("user_id") is None


def record_skill_run(
    session: Session, activity_events: Iterable[ActivityEvent], library: SkillLibrary
) -> list[str]:
    """Record actual terminal local runs without copying private prompts or outputs."""
    if not _local(session) or session.status not in {"done", "failed", "cancelled"}:
        return []
    events = [
        event
        for event in activity_events
        if event.session_id == session.id
        and event.kind == "tool_call.completed"
        and event.data.get("duration_ms") is not None
    ]
    if not any(
        event.data.get("name")
        not in {"skill_read", "skill_evidence", "skill_propose", "skill_apply"}
        for event in events
    ):
        return []
    observations = [
        {
            "event_id": event.id,
            "tool": event.data.get("name"),
            "is_error": bool(event.data.get("is_error")),
            "result_sha256": _digest(
                {
                    key: event.data.get(key)
                    for key in ("content_preview", "content_size", "metadata")
                }
            ),
        }
        for event in sorted(events, key=lambda item: (item.timestamp, item.id))
    ]
    generic = SkillLifecycle(session.cwd / ".harness/skills").save_evidence(
        {
            "session_id": session.id,
            "workspace": str(session.cwd),
            "skill": None,
            "revision": None,
            "status": session.status,
            "observations": observations,
            "conclusion": "execution recorded; outcome is not an independent correctness verdict",
        }
    )
    identifiers: list[str] = []
    active = session.metadata.get("active_skills", [])
    if not isinstance(active, list):
        return [generic]
    for name in sorted(set(n for n in active if isinstance(n, str))):
        if name not in library.skills:
            continue
        skill = library.get(name)
        lifecycle = SkillLifecycle(skill.directory.parent)
        revision = lifecycle.inspect(name)["revision"]
        activations = [
            event.data.get("metadata", {}).get("sha256")
            for event in events
            if isinstance(event.data.get("metadata"), dict)
            and event.data["metadata"].get("skill_activated") == name
        ]
        evidence = {
            "session_id": session.id,
            "skill": name,
            "revision": revision,
            "revision_observed": "at_recording",
            "activation_sha256": activations,
            "status": session.status,
            "observations": observations,
            "conclusion": "execution recorded; outcome is not an independent correctness verdict",
        }
        identifiers.append(lifecycle.save_evidence(evidence))
    return identifiers or [generic]


class _EvolutionTool:
    def __init__(self, session: Session, library: SkillLibrary, *, apply: bool) -> None:
        self.session = session
        self.library = library
        self.name = "skill_apply" if apply else "skill_propose"
        self.approval: ApprovalDecision = "prompt" if apply else "auto"
        self.effect_scope = "workspace_durable" if apply else "task_durable"
        self.description = (
            "Apply a stored skill proposal after approval. Include its exact full diff for review."
            if apply
            else "Propose an installed skill improvement, or create=true to propose a new workspace skill from recorded run evidence. Persists a reviewable diff; changes require skill_apply approval."
        )
        properties = (
            {
                "name": {"type": "string"},
                "proposal_id": {"type": "string"},
                "expected_base": {"type": "string"},
                "reviewed_diff": {"type": "string"},
            }
            if apply
            else {
                "name": {"type": "string"},
                "content": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "create": {
                    "type": "boolean",
                    "description": "Explicitly propose a new skill instead of updating an installed skill.",
                },
            }
        )
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": [key for key in properties if key != "create"],
            "additionalProperties": False,
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            if not _local(self.session):
                raise SkillError("skill evolution is available only to local sessions")
            args = call.arguments
            name = args.get("name")
            if not isinstance(name, str):
                raise SkillError("skill name is required")
            validate_skill_name(name)
            skill = self.library.skills.get(name)
            lifecycle = SkillLifecycle(
                skill.directory.parent if skill else self.session.cwd / ".harness/skills"
            )
            if self.name == "skill_apply":
                for key in ("proposal_id", "expected_base", "reviewed_diff"):
                    if not isinstance(args.get(key), str):
                        raise SkillError(f"{key} must be a string")
                proposal = lifecycle.proposal(args["proposal_id"])
                if proposal["name"] != name:
                    raise SkillError("proposal is for a different skill")
                result = lifecycle.apply(
                    args["proposal_id"],
                    expected=args["expected_base"],
                    reviewed_diff=args["reviewed_diff"],
                    session_id=self.session.id,
                )
                self.library.skills[name] = Skill.load(lifecycle.root / name)
            else:
                evidence = args.get("evidence_ids")
                if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
                    raise SkillError("evidence_ids must be a list of recorded evidence IDs")
                if not isinstance(args.get("content"), str) or not isinstance(
                    args.get("reason"), str
                ):
                    raise SkillError("content and reason must be strings")
                if type(args.get("create", False)) is not bool:
                    raise SkillError("create must be boolean")
                if args.get("create", False) and skill is not None:
                    raise SkillError("a skill with this name is already installed")
                result = lifecycle.propose(
                    name,
                    args["content"],
                    session_id=self.session.id,
                    evidence_ids=evidence,
                    reason=args["reason"],
                    create=args.get("create", False),
                )
            return ToolResult(tool_call_id=call.id, name=self.name, content=_json(result))
        except (SkillError, OSError, sqlite3.Error) as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)


class SkillEvidenceTool:
    name = "skill_evidence"
    description = "Read local execution evidence IDs/outcomes. Omit name for completed work that can inform a new skill; provide an installed skill name for its usage evidence. Evidence is observational, not a correctness verdict."
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"

    def __init__(self, session: Session, library: SkillLibrary) -> None:
        self.session = session
        self.library = library
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            name = call.arguments.get("name")
            if not _local(self.session) or (name is not None and not isinstance(name, str)):
                raise SkillError("a local session and optional installed skill name are required")
            root = (
                self.library.get(name).directory.parent
                if name is not None
                else self.session.cwd / ".harness/skills"
            )
            result = SkillLifecycle(root).evidence(name)[:20]
            return ToolResult(tool_call_id=call.id, name=self.name, content=_json(result))
        except (SkillError, OSError, sqlite3.Error) as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)


def skill_evolution_tools(session: Session, library: SkillLibrary) -> list[Tool]:
    if not _local(session):
        return []
    return [
        SkillEvidenceTool(session, library),
        _EvolutionTool(session, library, apply=False),
        _EvolutionTool(session, library, apply=True),
    ]
