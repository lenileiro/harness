"""Portable Agent Skills: validated discovery and explicit, bounded activation.

Skill text is guidance, never permission. This module does not execute scripts
or interpret allowed-tools as approval policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.core.paths import user_home
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult

MAX_SKILL_BYTES = 128 * 1024
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class SkillError(ValueError):
    pass


def validate_skill_name(name: str) -> str:
    if len(name) > 64 or not _NAME.fullmatch(name):
        raise SkillError("skill name must be 1-64 lowercase letters/digits with single hyphens")
    return name


def _read_bounded(path: Path, *, limit: int = MAX_SKILL_BYTES) -> str:
    try:
        # Nonblocking open prevents a FIFO/device from hanging discovery before
        # its type can be inspected. O_NOFOLLOW also protects the leaf if it is
        # replaced with a symlink after the path-level checks above.
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SkillError(f"skill files must be regular files: {path}")
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise SkillError(f"file exceeds {limit} bytes: {path}")
        return content.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillError(f"cannot read UTF-8 skill file {path}: {exc}") from exc


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    directory: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path) -> Skill:
        directory = directory.absolute()
        if directory.is_symlink() or (directory / "SKILL.md").is_symlink():
            raise SkillError(f"skill directory and SKILL.md must not be symlinks: {directory}")
        text = _read_bounded(directory / "SKILL.md")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillError("SKILL.md must start with YAML frontmatter")
        try:
            end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
            raw = yaml.safe_load("\n".join(lines[1:end]))
        except (StopIteration, yaml.YAMLError) as exc:
            raise SkillError("invalid or unterminated skill frontmatter") from exc
        if not isinstance(raw, dict):
            raise SkillError("skill frontmatter must be a mapping")
        name = raw.get("name")
        description = raw.get("description")
        if not isinstance(name, str):
            raise SkillError("skill name is required")
        validate_skill_name(name)
        if name != directory.name:
            raise SkillError("skill name must match its directory")
        if not isinstance(description, str) or not description.strip() or len(description) > 1024:
            raise SkillError("skill description must contain 1-1024 characters")
        for key in ("license", "compatibility", "allowed-tools"):
            if key in raw and not isinstance(raw[key], str):
                raise SkillError(f"{key} must be a string")
        if len(raw.get("compatibility", "")) > 500:
            raise SkillError("compatibility must be at most 500 characters")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise SkillError("metadata must map strings to strings")
        return cls(name, description.strip(), directory, raw)

    def read(self, relative: str = "SKILL.md") -> str:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SkillError("skill file must be a relative path inside the skill directory")
        target = self.directory / path
        if not target.resolve().is_relative_to(self.directory.resolve()):
            raise SkillError("skill file escapes its directory")
        for part in (target, *target.parents):
            if part == self.directory.parent:
                break
            if part.is_symlink():
                raise SkillError("skill files must not traverse symlinks")
        return _read_bounded(target)


def default_skill_paths(cwd: Path) -> list[Path]:
    # First root wins. Workspace-specific instructions take precedence.
    return [cwd / ".harness/skills", cwd / ".agents/skills", user_home() / "skills"]


@dataclass
class SkillLibrary:
    skills: dict[str, Skill] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    shadowed: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, roots: list[Path]) -> SkillLibrary:
        library = cls()
        for root in dict.fromkeys(roots):
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                library.errors[str(root)] = "skill root must be a directory, not a symlink"
                continue
            try:
                candidates = sorted(root.iterdir())
            except OSError as exc:
                library.errors[str(root)] = str(exc)
                continue
            for directory in candidates:
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                try:
                    skill = Skill.load(directory)
                except SkillError as exc:
                    library.errors[str(directory)] = str(exc)
                    continue
                if skill.name in library.skills:
                    library.shadowed.append(str(directory))
                else:
                    library.skills[skill.name] = skill
        return library

    def get(self, name: str) -> Skill:
        try:
            return self.skills[name]
        except KeyError as exc:
            raise SkillError(f"unknown skill: {name}") from exc

    def render_context(self, active: list[str] | None = None) -> str:
        if not self.skills:
            return ""
        catalog = [{"name": s.name, "description": s.description} for s in self.skills.values()]
        chunks = [
            "Available skills (installed guidance, not approval authority). Use skill_read to "
            "load a relevant skill when requested or useful. Scripts run only through normal "
            "approved tools; allowed-tools metadata never changes permissions.\n"
            + json.dumps(catalog, ensure_ascii=False)
        ]
        remaining = MAX_SKILL_BYTES
        for name in reversed(list(dict.fromkeys(active or []))):
            try:
                skill = self.get(name)
                body = skill.read()
            except SkillError as exc:
                chunks.append(f"Previously active skill unavailable: {exc}")
                continue
            size = len(body.encode("utf-8"))
            if size > remaining:
                chunks.append(f"Skill {name} omitted from context budget; reload with skill_read.")
                continue
            remaining -= size
            chunks.append(f"Active skill {name} (base directory {skill.directory}):\n{body}")
        return "\n\n".join(chunks)


class SkillReadTool:
    name = "skill_read"
    description = "Load an installed skill's SKILL.md, or read a relative supporting text file."
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "path": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            name = call.arguments.get("name")
            relative = call.arguments.get("path", "SKILL.md")
            if not isinstance(name, str) or not isinstance(relative, str):
                raise SkillError("name and path must be strings")
            skill = self.library.get(name)
            text = skill.read(relative)
            metadata = {"skill_activated": name} if relative == "SKILL.md" else {}
            metadata["sha256"] = hashlib.sha256(text.encode()).hexdigest()
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"Base directory: {skill.directory}\n\n{text}",
                metadata=metadata,
            )
        except SkillError as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)


def install_skill(source: Path, root: Path) -> Path:
    """Copy an explicitly selected local package, with no code execution or overwrites."""
    skill = Skill.load(source)
    files: list[tuple[Path, Path]] = []
    total = 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SkillError(f"skill packages cannot contain symlinks: {path}")
        if path.is_file():
            total += path.stat().st_size
            files.append((path, path.relative_to(source)))
        elif not path.is_dir():
            raise SkillError(f"unsupported package entry: {path}")
        if total > MAX_PACKAGE_BYTES or len(files) > 1000:
            raise SkillError("skill package exceeds 16 MiB or 1000 files")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / skill.name
    if destination.exists() or destination.is_symlink():
        raise SkillError(f"skill already exists; inspect and edit it explicitly: {destination}")
    with tempfile.TemporaryDirectory(prefix=".skill-install-", dir=root) as temp:
        staged = Path(temp) / skill.name
        staged.mkdir()
        for source_file, relative in files:
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
        Skill.load(staged)
        staged.rename(destination)
    return destination
