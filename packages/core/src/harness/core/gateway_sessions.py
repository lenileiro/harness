from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from harness.core.gateway_models import (
    GatewayRuntimeBinding,
    GatewaySessionBinding,
    GatewayUserProfile,
)
from harness.core.slug import slugify


def _slugify(value: str) -> str:
    return slugify(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GatewaySessionStore:
    def __init__(self, *, root: Path):
        self.root = root

    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    def ensure_layout(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conversation_lock(self, *, transport: str, user_id: str, thread_id: str) -> Iterator[bool]:
        """Serialize remote turns and decisions without blocking the async event loop."""
        key = hashlib.sha256(json.dumps([transport, user_id, thread_id]).encode()).hexdigest()
        path = self.root / "locks" / f"{key}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(path, timeout=0, mode=0o600)
        try:
            lock.acquire()
        except Timeout:
            yield False
            return
        try:
            yield True
        finally:
            lock.release()

    def _runtime_path(self, session_id: str) -> Path:
        key = hashlib.sha256(session_id.encode()).hexdigest()
        return self.root / "runtime-sessions" / f"{key}.json"

    def bind_runtime_session(self, binding: GatewayRuntimeBinding) -> None:
        previous = self.load_runtime_binding(binding.session_id)
        if previous is not None and (
            {key: value for key, value in previous.to_dict().items() if key != "max_steps"}
            != {key: value for key, value in binding.to_dict().items() if key != "max_steps"}
        ):
            raise ValueError("Gateway runtime session already has different ownership or settings")
        _write_json(self._runtime_path(binding.session_id), binding.to_dict())

    def load_runtime_binding(self, session_id: str) -> GatewayRuntimeBinding | None:
        path = self._runtime_path(session_id)
        if not path.is_file():
            return None
        binding = GatewayRuntimeBinding.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if binding.session_id != session_id:
            raise ValueError("Gateway runtime binding does not match its session")
        return binding

    def list_runtime_bindings(
        self, *, transport: str, user_id: str, thread_id: str
    ) -> list[GatewayRuntimeBinding]:
        items = []
        for path in sorted((self.root / "runtime-sessions").glob("*.json")):
            binding = GatewayRuntimeBinding.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if binding.belongs_to(transport=transport, user_id=user_id, thread_id=thread_id):
                items.append(binding)
        return items

    def new_id(self, transport: str, user_id: str, thread_id: str) -> str:
        title = f"{transport}-{user_id}-{thread_id}"
        return f"gw-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def save_session(self, session: GatewaySessionBinding) -> Path:
        self.ensure_layout()
        target = self.sessions_dir / session.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "session.json", session.to_dict())
        return target

    def load_session(self, session_id: str) -> GatewaySessionBinding:
        path = self.sessions_dir / session_id / "session.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return GatewaySessionBinding.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _profile_id(self, transport: str, user_id: str) -> str:
        identity = json.dumps([transport, user_id], separators=(",", ":"), ensure_ascii=False)
        return f"gwp-{hashlib.sha256(identity.encode()).hexdigest()}"

    def _profile_path(self, transport: str, user_id: str) -> Path:
        return self.profiles_dir / self._profile_id(transport, user_id) / "profile.json"

    def save_profile(self, profile: GatewayUserProfile) -> Path:
        self.ensure_layout()
        profile = replace(profile, id=self._profile_id(profile.transport, profile.user_id))
        target = self._profile_path(profile.transport, profile.user_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, profile.to_dict())
        return target.parent

    def load_profile(self, transport: str, user_id: str) -> GatewayUserProfile:
        path = self._profile_path(transport, user_id)
        if path.is_file():
            profile = GatewayUserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if profile.transport != transport or profile.user_id != user_id:
                raise ValueError("Gateway profile does not match its identity")
            return profile
        # Legacy slugs are ambiguous. Migrate only records whose embedded
        # transport and user match exactly; never infer ownership from a filename.
        candidates = [
            *sorted(self.profiles_dir.glob("*/profile.json")),
            *sorted(self.profiles_dir.glob("*.json")),
        ]
        for candidate in candidates:
            profile = GatewayUserProfile.from_dict(
                json.loads(candidate.read_text(encoding="utf-8"))
            )
            if profile.transport == transport and profile.user_id == user_id:
                profile = replace(profile, id=self._profile_id(transport, user_id))
                self.save_profile(profile)
                return profile
        raise FileNotFoundError(path)

    def list_profiles(self) -> list[GatewayUserProfile]:
        if not self.profiles_dir.exists():
            return []
        items: list[GatewayUserProfile] = []
        seen: set[tuple[str, str]] = set()
        paths = [
            *sorted(self.profiles_dir.glob("*/profile.json")),
            *sorted(self.profiles_dir.glob("*.json")),
        ]
        for path in paths:
            profile = GatewayUserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
            key = (profile.transport, profile.user_id)
            if key in seen:
                continue
            seen.add(key)
            canonical = self._profile_path(profile.transport, profile.user_id)
            if canonical.is_file():
                profile = self.load_profile(profile.transport, profile.user_id)
            items.append(profile)
        return items

    def get_or_create_profile(self, *, transport: str, user_id: str) -> GatewayUserProfile:
        try:
            return self.load_profile(transport, user_id)
        except FileNotFoundError:
            profile = GatewayUserProfile(
                id=self._profile_id(transport, user_id),
                transport=transport,
                user_id=user_id,
            )
            self.save_profile(profile)
            return profile

    def list_sessions(self) -> list[GatewaySessionBinding]:
        if not self.sessions_dir.exists():
            return []
        items: list[GatewaySessionBinding] = []
        for path in sorted(self.sessions_dir.iterdir()):
            payload = path / "session.json"
            if not payload.is_file():
                continue
            items.append(
                GatewaySessionBinding.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            )
        return items

    def list_user_sessions(self, *, transport: str, user_id: str) -> list[GatewaySessionBinding]:
        return [
            item
            for item in self.list_sessions()
            if item.transport == transport and item.user_id == user_id
        ]

    def get_or_create_session(
        self, *, transport: str, user_id: str, thread_id: str
    ) -> GatewaySessionBinding:
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(self.root / "sessions.lock", mode=0o600):
            return self._get_or_create_session(
                transport=transport, user_id=user_id, thread_id=thread_id
            )

    def _get_or_create_session(
        self, *, transport: str, user_id: str, thread_id: str
    ) -> GatewaySessionBinding:
        for item in self.list_sessions():
            if (
                item.transport == transport
                and item.user_id == user_id
                and item.thread_id == thread_id
            ):
                return item
        session = GatewaySessionBinding(
            id=self.new_id(transport, user_id, thread_id),
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
        )
        self.save_session(session)
        return session


__all__ = ["GatewaySessionStore"]
