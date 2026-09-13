"""Device authorization and rotating credentials for explicitly configured providers.

RFC 8628 polling observes the server interval. Refresh is serialized across
processes, persisted atomically, and quarantined after terminal/ambiguous errors.
No login runs implicitly during model inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.core import ConfigurationError
from harness.core.paths import read_regular_file, user_home


def validated_endpoint(value: str) -> str:
    url = httpx.URL(value)
    if (
        (
            url.scheme != "https"
            and not (url.scheme == "http" and url.host in {"127.0.0.1", "localhost", "::1"})
        )
        or not url.host
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Account endpoints require HTTPS without credentials, query, or fragment; HTTP is allowed on loopback"
        )
    return str(url)


class OAuthAccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    device_authorization_endpoint: str
    token_endpoint: str
    client_id: str = Field(min_length=1, max_length=256)
    scopes: str = ""
    refresh_token_header: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
    timeout: float = Field(default=20, gt=0, le=120)

    @model_validator(mode="after")
    def endpoints(self):
        validated_endpoint(self.device_authorization_endpoint)
        validated_endpoint(self.token_endpoint)
        return self


class AccountAuth:
    def __init__(
        self,
        name: str,
        config: OAuthAccountConfig,
        *,
        resource: str,
        home: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.name, self.config = name, config
        self.resource = validated_endpoint(resource)
        self.transport = transport
        binding = {"name": name, "config": config.model_dump(), "resource": self.resource}
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.path = (home or user_home()) / "auth/accounts" / f"{key}.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        value = json.loads(read_regular_file(self.path, max_bytes=1024 * 1024))
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ConfigurationError("Account credential file is invalid; log in again")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump({"version": 1, **value}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _tokens(self, payload: Any, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("access_token"), str)
            or not payload["access_token"]
            or not isinstance(payload.get("token_type", "Bearer"), str)
            or payload.get("token_type", "Bearer").casefold() != "bearer"
        ):
            raise ConfigurationError("Account server returned invalid credentials")
        ttl = payload.get("expires_in")
        if (
            not isinstance(ttl, (int, float))
            or isinstance(ttl, bool)
            or not math.isfinite(ttl)
            or ttl <= 0
            or ttl > 365 * 86400
        ):
            raise ConfigurationError("Account server returned invalid token lifetime")
        refresh = payload.get("refresh_token") or (previous or {}).get("refresh_token")
        if refresh is not None and (not isinstance(refresh, str) or not refresh):
            raise ConfigurationError("Account server returned invalid refresh credentials")
        return {
            "version": 1,
            "access_token": payload["access_token"],
            "refresh_token": refresh,
            "expires_at": time.time() + ttl,
            "scope": payload.get("scope", self.config.scopes),
            "quarantined": False,
        }

    def _post(self, endpoint: str, *, data: dict, headers: dict | None = None) -> tuple[int, dict]:
        try:
            with (
                httpx.Client(
                    timeout=self.config.timeout,
                    transport=self.transport,
                    follow_redirects=False,
                    trust_env=False,
                ) as client,
                client.stream(
                    "POST",
                    endpoint,
                    data=data,
                    headers={"Accept": "application/json", **(headers or {})},
                ) as response,
            ):
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 1024 * 1024:
                        raise ConfigurationError("Account response exceeds its size limit")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("expected object")
                return response.status_code, payload
        except (httpx.HTTPError, ValueError) as exc:
            raise ConfigurationError(
                f"Account transport failed ({type(exc).__name__}); credential values were not logged"
            ) from None

    async def login(
        self,
        display: Callable[[str, str], Awaitable[None]],
        *,
        sleep=asyncio.sleep,
        monotonic=time.monotonic,
    ) -> None:
        status, payload = await asyncio.to_thread(
            self._post,
            self.config.device_authorization_endpoint,
            data={"client_id": self.config.client_id, "scope": self.config.scopes},
        )
        required = ("device_code", "user_code", "verification_uri")
        if status != 200 or any(
            not isinstance(payload.get(key), str) or not payload[key] for key in required
        ):
            raise ConfigurationError(f"Device authorization failed (HTTP {status})")
        uri = payload.get("verification_uri_complete", payload["verification_uri"])
        parsed = httpx.URL(uri)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ConfigurationError("Account verification link must use HTTPS without credentials")
        ttl, interval = payload.get("expires_in"), payload.get("interval", 5)
        if (
            not isinstance(ttl, (int, float))
            or isinstance(ttl, bool)
            or not 0 < ttl <= 3600
            or not isinstance(interval, (int, float))
            or isinstance(interval, bool)
            or not 0 < interval <= 60
        ):
            raise ConfigurationError(
                "Device authorization returned invalid expiry or polling interval"
            )
        await display(str(uri), payload["user_code"])
        deadline = monotonic() + ttl
        while monotonic() < deadline:
            await sleep(min(interval, max(0, deadline - monotonic())))
            if monotonic() >= deadline:
                break
            status, result = await asyncio.to_thread(
                self._post,
                self.config.token_endpoint,
                data={
                    "client_id": self.config.client_id,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": payload["device_code"],
                },
            )
            if status == 200:
                tokens = self._tokens(result)
                await asyncio.to_thread(self._store_login, tokens)
                return
            error = result.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error in {"access_denied", "expired_token"}:
                raise ConfigurationError(f"Account authorization {error}; run accounts login again")
            raise ConfigurationError(f"Account authorization failed (HTTP {status})")
        raise ConfigurationError("Device authorization expired; run accounts login again")

    def _store_login(self, tokens: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.path.with_suffix(".lock"), timeout=30, mode=0o600):
            self._write(tokens)

    def _access_token(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.path.with_suffix(".lock"), timeout=150, mode=0o600):
            state = self._read()
            if not state or state.get("quarantined"):
                raise ConfigurationError(
                    f"Account login required: harness accounts login {self.name}"
                )
            if state.get("expires_at", 0) > time.time() + 30:
                return state["access_token"]
            refresh = state.get("refresh_token")
            if not refresh:
                raise ConfigurationError(f"Account expired: harness accounts login {self.name}")
            data = {"client_id": self.config.client_id, "grant_type": "refresh_token"}
            headers = {}
            if self.config.refresh_token_header:
                headers[self.config.refresh_token_header] = refresh
            else:
                data["refresh_token"] = refresh
            try:
                status, payload = self._post(self.config.token_endpoint, data=data, headers=headers)
                if status in {429, 500, 502, 503, 504}:
                    raise TimeoutError(
                        "Account refresh is temporarily unavailable; credentials retained"
                    )
                if status != 200:
                    raise ConfigurationError("Account refresh rejected; log in again")
                updated = self._tokens(payload, state)
            except ConfigurationError:
                self._write({**state, "quarantined": True, "refresh_token": None})
                raise
            self._write(updated)
            return updated["access_token"]

    async def access_token(self) -> str:
        return await asyncio.to_thread(self._access_token)

    def status(self) -> dict[str, Any]:
        state = self._read()
        return {
            "name": self.name,
            "configured": bool(state),
            "expired": bool(state) and state.get("expires_at", 0) <= time.time(),
            "quarantined": bool(state.get("quarantined")),
            "expires_at": state.get("expires_at"),
        }

    def logout(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.path.with_suffix(".lock"), timeout=30, mode=0o600):
            self.path.unlink(missing_ok=True)
