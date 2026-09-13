"""Pinned matrix-nio crypto with explicit device trust and Harness-owned IO."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import httpx
from filelock import FileLock, Timeout
from nio import (
    AsyncClient,
    AsyncClientConfig,
    ErrorResponse,
    KeysQueryResponse,
    MegolmEvent,
    SyncResponse,
)
from nio.crypto import ENCRYPTION_ENABLED, decrypt_attachment, encrypt_attachment
from nio.exceptions import EncryptionError, LocalProtocolError

from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.core.gateway_channels import ChannelStore


class NioHTTPClient(AsyncClient):
    """Keep SDK crypto/state machines while enforcing Harness HTTP boundaries.

    matrix-nio 0.26.0's high-level methods delegate JSON requests to _send.
    This adapter deliberately supports only those requests, with no retries,
    token-bearing query strings, redirect following, or plaintext room sends.
    """

    transport: Transport
    raw_sync: dict[str, Any]

    async def _send(
        self,
        response_class: type,
        method: str,
        path: str,
        data: Any = None,
        response_data: tuple[Any, ...] | None = None,
        content_type: str | None = None,
        trace_context: Any = None,
        data_provider: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109 - pinned SDK override signature
        content_length: int | None = None,
        save_to: os.PathLike | None = None,
    ):
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/_matrix/"):
            raise ChannelError("Matrix SDK requested an unexpected endpoint")
        decoded_path = unquote(parsed.path)
        if "/send/" in decoded_path and "/send/m.room.encrypted/" not in decoded_path:
            raise ChannelError("Matrix E2EE refused a plaintext room send")
        if data_provider or save_to or not (data is None or isinstance(data, str)):
            raise ChannelError("Unsupported Matrix SDK IO operation")
        clean_path = urlunsplit(
            (
                "",
                "",
                parsed.path,
                urlencode(
                    [
                        (key, value)
                        for key, value in parse_qsl(parsed.query)
                        if key != "access_token"
                    ]
                ),
                "",
            )
        )
        try:
            async with self.transport.client.stream(
                method,
                self.homeserver + clean_path,
                content=data,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
                follow_redirects=False,
                timeout=45,
            ) as response:
                if response.status_code == 429:
                    raise RateLimited(float(response.headers.get("Retry-After", "1")))
                if not response.is_success:
                    raise ChannelError(
                        f"Matrix E2EE request rejected (HTTP {response.status_code}); no automatic retry"
                    )
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 24 * 1024 * 1024:
                        raise ChannelError("Matrix sync/key response exceeds 24 MiB")
            payload = json.loads(raw)
            original_sync = copy.deepcopy(payload) if response_class is SyncResponse else None
            result = response_class.from_dict(payload, *(response_data or ()))
            if isinstance(result, ErrorResponse):
                raise ChannelError("Matrix E2EE response failed SDK validation")
            if isinstance(result, SyncResponse):
                assert original_sync is not None
                self.raw_sync = original_sync
            await self.receive_response(result)
            return result
        except httpx.HTTPError:
            raise ChannelError(
                "Matrix E2EE connection interrupted; inspect delivery outcome before retrying"
            ) from None
        except (ValueError, TypeError, KeyError):
            raise ChannelError("Matrix E2EE returned an invalid response") from None


def _private_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ChannelError("matrix_store_path must be an absolute private directory")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        if component in {".", ".."}:
            raise ChannelError("Matrix crypto store must not contain traversal components")
        current = current / component
        if current.is_symlink():
            raise ChannelError("Matrix crypto store path must not traverse symlinks")
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise ChannelError("Matrix crypto store must be a directory")
    if os.name != "nt" and (path.stat().st_mode & 0o077):
        raise ChannelError("Matrix crypto store directory must be private (chmod 700)")


class MatrixCrypto:
    def __init__(self, transport: Transport, homeserver: str, user_id: str, device_id: str):
        if not ENCRYPTION_ENABLED:
            raise ChannelError(
                "Matrix E2EE requires the CLI matrix extra (matrix-nio[e2e] and vodozemac)"
            )
        config = transport.config
        if not device_id or device_id != config.matrix_device_id:
            raise ChannelError("Matrix token device ID does not match matrix_device_id")
        if config.matrix_unverified_policy != "reject":
            raise ChannelError("Matrix E2EE requires the reject policy for unverified devices")
        variable = config.matrix_pickle_key_env
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ChannelError("Matrix E2EE requires a pickle-key environment reference")
        passphrase = os.environ.get(variable, "")
        if len(passphrase) < 32:
            raise ChannelError(
                "Matrix pickle-key environment reference is missing or shorter than 32 characters"
            )
        transport.secrets.append(passphrase)
        root = Path(config.matrix_store_path).expanduser()
        _private_directory(root)
        identity = json.dumps(
            [homeserver, user_id, device_id, config.profile], separators=(",", ":")
        )
        directory = root / hashlib.sha256(identity.encode()).hexdigest()
        _private_directory(directory)
        for file in directory.iterdir():
            mode = file.lstat().st_mode
            if not stat.S_ISREG(mode) or file.is_symlink() or (os.name != "nt" and mode & 0o077):
                raise ChannelError("Matrix crypto store contains an unsafe or nonprivate file")
        self.lock = FileLock(directory / "device.lock", timeout=0, mode=0o600)
        try:
            self.lock.acquire()
        except Timeout:
            raise ChannelError("This Matrix device store is already in use") from None
        self.transport = transport
        self.ready = asyncio.Event()
        self.operation = asyncio.Lock()
        self.client: NioHTTPClient | None = None
        self.directory = directory
        self._unresolved: list[dict[str, Any]] = []
        try:
            database = directory / "crypto.db"
            fd = os.open(database, os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            os.close(fd)
            client = NioHTTPClient(
                homeserver,
                user_id,
                device_id=device_id,
                store_path=str(directory),
                config=AsyncClientConfig(
                    encryption_enabled=True,
                    store_name="crypto.db",
                    pickle_key=passphrase,
                    store_sync_tokens=False,
                    max_limit_exceeded=0,
                    max_timeouts=0,
                ),
            )
            client.transport = transport
            client.raw_sync = {}
            self.client = client
            client.restore_login(user_id, device_id, transport.token)
            if client.olm is None or client.store is None:
                raise ChannelError("Matrix SDK did not initialize its persistent crypto store")
            self.apply_trust()
        except BaseException:
            if self.client is not None and self.client.store is not None:
                self.client.store.database.close()
            self.lock.release()
            raise

    def apply_trust(self) -> None:
        client = self.client
        assert client is not None and client.olm is not None
        pins = self.transport.config.matrix_trusted_devices
        for device in client.device_store:
            if device.user_id == client.user_id and device.id == client.device_id:
                continue
            expected = pins.get(device.user_id, {}).get(device.id, "").replace(" ", "")
            trusted = bool(expected) and hmac.compare_digest(
                expected.encode(), device.ed25519.encode()
            )
            # Remove old grants before applying the current configuration.
            client.unignore_device(device)
            client.unblacklist_device(device)
            if trusted:
                client.verify_device(device)
            else:
                client.unverify_device(device)

    async def initialize_keys(self) -> None:
        client = self.client
        assert client is not None and client.olm is not None
        client.olm.users_for_key_query.add(client.user_id)
        response = await client.keys_query()
        if not isinstance(response, KeysQueryResponse):
            raise ChannelError("Matrix identity-key lookup failed")
        if response.failures:
            raise ChannelError(
                "Matrix identity-key lookup was incomplete; refusing to replace device keys"
            )
        own = response.device_keys.get(client.user_id, {}).get(client.device_id, {})
        keys = own.get("keys", {})
        if keys and any(
            keys.get(f"{kind}:{client.device_id}") != value
            for kind, value in client.olm.account.identity_keys.items()
        ):
            raise ChannelError(
                "Matrix server device keys differ from the local store; restore the original store or use a new device/token"
            )
        if client.should_upload_keys:
            await client.keys_upload()

    async def close(self) -> None:
        client, self.client = self.client, None
        try:
            if client is not None:
                await client.close()
                if client.store is not None:
                    client.store.database.close()
        finally:
            self.lock.release()

    async def sync(self, store: ChannelStore) -> dict[str, Any]:
        async with self.operation:
            client = self.client
            assert client is not None and client.olm is not None
            first = not self.ready.is_set()
            since = store.get("since")
            await client.sync(timeout=0 if first else 30000, since=since, full_state=first)
            if client.should_query_keys:
                await client.keys_query()
            self.apply_trust()
            if client.should_upload_keys:
                await client.keys_upload()
            raw = client.raw_sync
            pending = store.get("matrix_pending_ciphertext", [])
            for room_id, room in raw.get("rooms", {}).get("join", {}).items():
                room["timeline"]["events"] = [
                    event
                    for event in room.get("timeline", {}).get("events", [])
                    if event.get("type") == "m.room.encrypted"
                ]
                if since:
                    pending.extend(
                        {"room_id": room_id, "event": event}
                        for event in room["timeline"]["events"]
                        if event.get("sender") in self.transport.config.allowed_users
                        and event.get("sender") != client.user_id
                    )
                room["timeline"]["events"] = []
            retained = {}
            for entry in pending:
                room_id, source = entry["room_id"], entry["event"]
                if (
                    self.transport.config.allowed_channels
                    and room_id not in self.transport.config.allowed_channels
                ):
                    continue
                if source.get("sender") not in self.transport.config.allowed_users:
                    continue
                key = str(source.get("event_id", ""))
                try:
                    encrypted = MegolmEvent.from_dict({**source, "room_id": room_id})
                    if not isinstance(encrypted, MegolmEvent):
                        continue
                    event = client.decrypt_event(encrypted)
                    if not event.decrypted or not event.verified:
                        retained[key] = entry
                        continue
                except (EncryptionError, LocalProtocolError):
                    retained[key] = entry
                    continue
                room = (
                    raw.setdefault("rooms", {})
                    .setdefault("join", {})
                    .setdefault(room_id, {"timeline": {"events": []}})
                )
                room["timeline"]["events"].append(event.source)
            if len(retained) > 256 or len(json.dumps(retained).encode()) > 16 * 1024 * 1024:
                raise ChannelError(
                    "Matrix pending encrypted events exceed the limit; restore missing keys or review device trust before continuing"
                )
            # Keep every ciphertext durable until MatrixTransport has ingested
            # decrypted messages; a crash may replay IDs but cannot lose them.
            durable = {str(entry["event"].get("event_id", "")): entry for entry in pending}
            if len(durable) > 256 or len(json.dumps(durable).encode()) > 16 * 1024 * 1024:
                raise ChannelError("Matrix encrypted inbox exceeds its bounded pending capacity")
            store.set("matrix_pending_ciphertext", list(durable.values()))
            self._unresolved = list(retained.values())
            store.set(
                "matrix_encryption",
                {
                    "verified_only": True,
                    "pending_events": len(retained),
                    "device_id": client.device_id,
                    "fingerprint": client.olm.account.identity_keys["ed25519"],
                },
            )
            self.ready.set()
            return raw

    def commit_sync(self, store: ChannelStore) -> None:
        store.set("matrix_pending_ciphertext", self._unresolved)

    async def ensure_room(self, room_id: str) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=45)
        client = self.client
        assert client is not None and client.olm is not None
        room = client.rooms.get(room_id)
        if room is None or not room.encrypted:
            raise ChannelError(
                "Matrix E2EE requires a synced encrypted room; plaintext fallback is forbidden"
            )
        await client.joined_members(room_id)
        # Refresh devices before every send, including already-shared sessions.
        client.olm.users_for_key_query.update(room.users)
        response = await client.keys_query()
        if not isinstance(response, KeysQueryResponse) or response.failures:
            raise ChannelError("Matrix device-key lookup was incomplete; message was not sent")
        self.apply_trust()
        for user_id in room.users:
            reported = response.device_keys.get(user_id, {})
            if user_id != client.user_id and not reported:
                raise ChannelError(
                    "Matrix room member has no usable encryption devices; message was not sent"
                )
            for device_id, info in reported.items():
                if user_id == client.user_id and device_id == client.device_id:
                    continue
                device = client.device_store[user_id].get(device_id)
                if (
                    device is None
                    or info.get("keys", {}).get(f"ed25519:{device_id}") != device.ed25519
                ):
                    raise ChannelError(
                        "Matrix device keys changed or failed verification; message was not sent"
                    )
            for device in client.device_store.active_user_devices(user_id):
                if user_id == client.user_id and device.id == client.device_id:
                    continue
                if not client.olm.is_device_verified(device):
                    raise ChannelError(
                        "Matrix room has unverified devices; pin every intended device fingerprint before sending"
                    )

    async def check_room(self, room_id: str) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=45)
        async with self.operation:
            await self.ensure_room(room_id)

    async def send(self, room_id: str, content: dict[str, Any], delivery_id: str) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=45)
        async with self.operation:
            await self.ensure_room(room_id)
            client = self.client
            assert client is not None and client.olm is not None
            try:
                if client.olm.should_share_group_session(room_id):
                    await client.share_group_session(room_id, ignore_unverified_devices=False)
                    session = client.olm.outbound_group_sessions[room_id]
                    expected = {
                        (user, device.id)
                        for user in client.rooms[room_id].users
                        for device in client.device_store.active_user_devices(user)
                        if not (user == client.user_id and device.id == client.device_id)
                    }
                    if not expected.issubset(session.users_shared_with):
                        client.invalidate_outbound_session(room_id)
                        raise ChannelError(
                            "Matrix group-key delivery was incomplete; message was not sent"
                        )
                await client.room_send(
                    room_id,
                    "m.room.message",
                    content,
                    tx_id=delivery_id,
                    ignore_unverified_devices=False,
                )
            except (EncryptionError, LocalProtocolError):
                raise ChannelError(
                    "Matrix encryption or device verification failed; plaintext fallback is forbidden"
                ) from None

    @staticmethod
    def encrypt_media(raw: bytes) -> tuple[bytes, dict[str, Any]]:
        return encrypt_attachment(raw)

    @staticmethod
    def decrypt_media(raw: bytes, file: dict[str, Any]) -> bytes:
        try:
            if (
                file.get("v") != "v2"
                or file["key"]["alg"] != "A256CTR"
                or file["key"]["kty"] != "oct"
            ):
                raise ValueError
            return decrypt_attachment(raw, file["key"]["k"], file["hashes"]["sha256"], file["iv"])
        except (ValueError, KeyError, TypeError, EncryptionError):
            raise ChannelError("Matrix encrypted attachment integrity check failed") from None
