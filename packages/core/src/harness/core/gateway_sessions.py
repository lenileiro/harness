from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from harness.core.gateway_models import GatewaySessionBinding, GatewayUserProfile


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "item"


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

    def new_id(self, transport: str, user_id: str, thread_id: str) -> str:
        title = f"{transport}-{user_id}-{thread_id}"
        return f"gw-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def save_session(self, session: GatewaySessionBinding) -> Path:
        self.ensure_layout()
        target = self.sessions_dir / session.id
        target.mkdir(parents=True, exist_ok=True)
        (target / "session.json").write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8",
        )
        return target

    def load_session(self, session_id: str) -> GatewaySessionBinding:
        path = self.sessions_dir / session_id / "session.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return GatewaySessionBinding.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _profile_id(self, transport: str, user_id: str) -> str:
        return f"gwp-{_slugify(f'{transport}-{user_id}')[:48]}"

    def _profile_path(self, transport: str, user_id: str) -> Path:
        return self.profiles_dir / self._profile_id(transport, user_id) / "profile.json"

    def _legacy_profile_path(self, transport: str, user_id: str) -> Path:
        name = f"{_slugify(transport)}-{_slugify(user_id)}"
        return self.profiles_dir / f"{name}.json"

    def save_profile(self, profile: GatewayUserProfile) -> Path:
        self.ensure_layout()
        target = self._profile_path(profile.transport, profile.user_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return target.parent

    def load_profile(self, transport: str, user_id: str) -> GatewayUserProfile:
        path = self._profile_path(transport, user_id)
        if not path.is_file():
            legacy = self._legacy_profile_path(transport, user_id)
            if legacy.is_file():
                path = legacy
            else:
                for candidate in sorted(self.profiles_dir.glob("*/profile.json")):
                    profile = GatewayUserProfile.from_dict(
                        json.loads(candidate.read_text(encoding="utf-8"))
                    )
                    if profile.transport == transport and profile.user_id == user_id:
                        return profile
                raise FileNotFoundError(path)
        return GatewayUserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

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
