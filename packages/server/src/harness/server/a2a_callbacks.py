"""Owned, HMAC-signed A2A callbacks with durable notification retries."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import httpx
from a2a.compat.v0_3.conversions import to_compat_task
from a2a.types import a2a_pb2 as proto
from a2a.utils.errors import (
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotFoundError,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from harness.server.service import HarnessService


def callback_url(value: str) -> str:
    try:
        url = httpx.URL(value)
        if (
            url.scheme != "https"
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
            or len(value) > 2048
        ):
            raise ValueError
        with suppress(ValueError):
            if not ipaddress.ip_address(url.host).is_global:
                raise InvalidParamsError("Callback destinations must use public HTTPS addresses")
        return str(url)
    except (ValueError, httpx.InvalidURL):
        raise InvalidParamsError(
            "Callback URL must be HTTPS without credentials, query, or fragment"
        ) from None


class CallbackGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    owner: str = Field(min_length=1, max_length=512)
    url: str
    secret_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    bearer_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def validate_url(self):
        try:
            callback_url(self.url)
        except InvalidParamsError:
            raise ValueError(
                "Callback URL must be public HTTPS without credentials, query, or fragment"
            ) from None
        return self


async def resolve_public(host: str, port: int) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        records = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        return list({str(record[4][0]) for record in records})


class CallbackHTTP:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Callable[[str, int], Awaitable[list[str]]] = resolve_public,
    ):
        self.transport = transport
        self.resolver = resolver

    async def post(self, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, str]:
        original = httpx.URL(callback_url(url))
        async with asyncio.timeout(15):
            addresses = await self.resolver(original.host, original.port or 443)
            if not addresses:
                raise InvalidParamsError("Callback DNS returned no usable addresses")
            parsed = [ipaddress.ip_address(address) for address in addresses]
            if any(
                not address.is_global
                or address.is_multicast
                or (
                    isinstance(address, ipaddress.IPv6Address)
                    and address.ipv4_mapped is not None
                    and not address.ipv4_mapped.is_global
                )
                for address in parsed
            ):
                raise InvalidParamsError("Callback DNS resolved to a prohibited address")
            selected = str(sorted(parsed, key=lambda address: (address.version, int(address)))[0])
            pinned = original.copy_with(host=selected)
            authority = original.netloc.decode("ascii")
            # Connect to the validated numeric address. Preserve Host and TLS
            # hostname verification; a second DNS lookup cannot change the peer.
            async with (
                httpx.AsyncClient(
                    transport=self.transport, trust_env=False, follow_redirects=False, timeout=10
                ) as client,
                client.stream(
                    "POST",
                    pinned,
                    content=body,
                    headers={**headers, "Host": authority},
                    extensions={"sni_hostname": original.host},
                ) as response,
            ):
                return response.status_code, response.headers.get("Retry-After", "")


class CallbackManager:
    def __init__(
        self,
        service: HarnessService,
        grants: list[CallbackGrant],
        *,
        http: CallbackHTTP | None = None,
        environment: Mapping[str, str] | None = None,
        max_attempts: int = 5,
    ):
        if not grants or len(grants) > 100 or not 1 <= max_attempts <= 10:
            raise ValueError("Configure 1..100 callback grants and 1..10 delivery attempts")
        self.service = service
        self.http = http or CallbackHTTP()
        self.max_attempts = max_attempts
        self.grants: dict[tuple[str, str], tuple[str, str]] = {}
        values = environment if environment is not None else os.environ
        for grant in grants:
            secret = values.get(grant.secret_env, "")
            bearer = values.get(grant.bearer_env, "") if grant.bearer_env else ""
            if len(secret) < 32 or (
                grant.bearer_env and not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", bearer)
            ):
                raise ValueError(
                    "Callback signing or bearer environment reference is missing or invalid"
                )
            key = (grant.owner, callback_url(grant.url))
            if key in self.grants:
                raise ValueError("Callback grants must have distinct caller/URL pairs")
            self.grants[key] = (secret, bearer)
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.last_error: str | None = None

    def prepare(
        self, owner: str, value: proto.TaskPushNotificationConfig, task_id: str
    ) -> proto.TaskPushNotificationConfig:
        from harness.server.service import identifier

        normalized = callback_url(value.url)
        if (owner, normalized) not in self.grants:
            raise InvalidParamsError("Callback endpoint is not explicitly granted to this caller")
        if value.tenant or (value.task_id and value.task_id != task_id):
            raise InvalidParamsError("Callback task or tenant does not match the owned task")
        if value.id and (len(value.id) > 128 or not re.fullmatch(r"[A-Za-z0-9_.-]+", value.id)):
            raise InvalidParamsError(
                "Callback configuration ID must be 1..128 simple identifier characters"
            )
        if (
            value.token != value.token.strip()
            or len(value.token) > 512
            or any(ord(character) < 32 or ord(character) > 126 for character in value.token)
        ):
            raise InvalidParamsError(
                "Callback correlation token must contain at most 512 printable ASCII characters"
            )
        bearer = self.grants[(owner, normalized)][1]
        scheme = value.authentication.scheme.lower()
        if (
            value.authentication.credentials
            or scheme not in {"", "hmac-sha256", "bearer"}
            or (scheme == "bearer" and not bearer)
        ):
            raise InvalidParamsError(
                "Callback authentication must use operator-configured HMAC or bearer references; inline credentials are unsupported"
            )
        return proto.TaskPushNotificationConfig(
            id=value.id or identifier("callback"),
            task_id=task_id,
            url=normalized,
            token=value.token,
            authentication=proto.AuthenticationInfo(scheme="Bearer" if bearer else "HMAC-SHA256"),
        )

    async def owned(self, owner: str, task_id: str) -> None:
        if not await self.service.store.rows(
            "SELECT id FROM api_a2a_tasks WHERE id=? AND owner=?", (task_id, owner)
        ):
            raise TaskNotFoundError()

    async def create(
        self, owner: str, value: proto.TaskPushNotificationConfig, *, protocol: str = "1.0"
    ) -> proto.TaskPushNotificationConfig:
        if protocol not in {"0.3", "1.0"}:
            raise InvalidParamsError("Unsupported callback protocol")
        await self.owned(owner, value.task_id)
        config = self.prepare(owner, value, value.task_id)
        async with self.service.store.connection() as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                "SELECT COUNT(*) FROM api_a2a_callbacks WHERE owner=? AND active=1", (owner,)
            ) as cursor:
                count = await cursor.fetchone()
            async with db.execute(
                "SELECT config,active,protocol FROM api_a2a_callbacks WHERE owner=? AND task_id=? AND id=?",
                (owner, config.task_id, config.id),
            ) as cursor:
                existing = await cursor.fetchone()
            serialized = json.dumps(MessageToDict(config), sort_keys=True)
            if existing:
                if existing[0] != serialized or not existing[1] or existing[2] != protocol:
                    raise InvalidParamsError(
                        "Callback ID already has another configuration; delete it and choose a new ID"
                    )
            else:
                if count is not None and count[0] >= 100:
                    raise InvalidParamsError(
                        "Caller callback limit reached (100 active subscriptions)"
                    )
                await db.execute(
                    "INSERT INTO api_a2a_callbacks (owner,task_id,id,config,last_seq,active,protocol) VALUES (?,?,?,?,0,1,?)",
                    (owner, config.task_id, config.id, serialized, protocol),
                )
            await db.commit()
        return config

    async def get(
        self, owner: str, task_id: str, config_id: str, *, protocol: str = "1.0"
    ) -> proto.TaskPushNotificationConfig:
        await self.owned(owner, task_id)
        rows = await self.service.store.rows(
            "SELECT config,protocol FROM api_a2a_callbacks WHERE owner=? AND task_id=? AND id=? AND active=1",
            (owner, task_id, config_id),
        )
        if not rows:
            raise InvalidParamsError("Callback configuration not found")
        if rows[0]["protocol"] != protocol:
            raise InvalidParamsError(
                "Use the A2A version originally used to register this callback"
            )
        return ParseDict(json.loads(rows[0]["config"]), proto.TaskPushNotificationConfig())

    async def list(
        self, owner: str, task_id: str, *, protocol: str = "1.0"
    ) -> list[proto.TaskPushNotificationConfig]:
        await self.owned(owner, task_id)
        rows = await self.service.store.rows(
            "SELECT config FROM api_a2a_callbacks WHERE owner=? AND task_id=? AND active=1 AND protocol=? ORDER BY id LIMIT 100",
            (owner, task_id, protocol),
        )
        return [
            ParseDict(json.loads(row["config"]), proto.TaskPushNotificationConfig()) for row in rows
        ]

    async def delete(
        self, owner: str, task_id: str, config_id: str, *, protocol: str = "1.0"
    ) -> None:
        await self.get(owner, task_id, config_id, protocol=protocol)
        async with self.service.store.connection() as db:
            await db.execute(
                "UPDATE api_a2a_callbacks SET active=0 WHERE owner=? AND task_id=? AND id=?",
                (owner, task_id, config_id),
            )
            await db.execute(
                "UPDATE api_a2a_callback_deliveries SET state='cancelled' WHERE owner=? AND task_id=? AND config_id=? AND state='pending'",
                (owner, task_id, config_id),
            )
            await db.commit()

    async def deliveries(self, owner: str) -> list[dict[str, Any]]:
        return await self.service.store.rows(
            "SELECT id,task_id,config_id,event_seq,state,attempts,next_attempt,error FROM api_a2a_callback_deliveries WHERE owner=? ORDER BY event_seq DESC LIMIT 100",
            (owner,),
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        async with self.service.store.connection() as db:
            await db.execute(
                "UPDATE api_a2a_callback_deliveries SET state='pending',error='Interrupted notification attempt; receiver must deduplicate delivery ID' WHERE state='sending'"
            )
            await db.execute(
                "UPDATE api_a2a_callback_deliveries SET state='cancelled' WHERE state='pending' AND NOT EXISTS (SELECT 1 FROM api_a2a_callbacks c WHERE c.owner=api_a2a_callback_deliveries.owner AND c.task_id=api_a2a_callback_deliveries.task_id AND c.id=api_a2a_callback_deliveries.config_id AND c.active=1)"
            )
            await db.commit()
        self._task = asyncio.create_task(self.run(), name="harness-a2a-callbacks")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def collect(self) -> None:
        from harness.server.a2a import STATES, parts
        from harness.server.service import identifier

        configs = await self.service.store.rows("SELECT * FROM api_a2a_callbacks WHERE active=1")
        for config in configs:
            events = await self.service.store.rows(
                "SELECT e.seq,e.payload,e.created_at,r.id AS run_id,r.session_id FROM api_events e JOIN api_a2a_task_runs t ON t.run_id=e.run_id JOIN api_runs r ON r.id=e.run_id WHERE t.task_id=? AND r.owner=? AND e.seq>? ORDER BY e.seq LIMIT 200",
                (config["task_id"], config["owner"], config["last_seq"]),
            )
            queued = []
            for event in events:
                value = json.loads(event["payload"])
                if value.get("type") != "run_status" or value.get("state") not in STATES:
                    continue
                delivery_id = identifier("notification")
                task = proto.Task(
                    id=config["task_id"],
                    context_id=event["session_id"],
                    status=proto.TaskStatus(state=STATES[value["state"]]),
                    metadata={
                        "harness_run_id": event["run_id"],
                        "harness_delivery_id": delivery_id,
                        "harness_event_sequence": event["seq"],
                    },
                )
                task.status.timestamp.FromJsonString(event["created_at"])
                if value.get("error"):
                    task.status.message.CopyFrom(
                        proto.Message(
                            message_id=f"{event['run_id']}-status",
                            role=proto.ROLE_AGENT,
                            parts=[proto.Part(text=value["error"])],
                        )
                    )
                if value["state"] == "completed":
                    done = await self.service.store.rows(
                        "SELECT payload FROM api_events WHERE run_id=? AND json_extract(payload,'$.type')='done' ORDER BY seq DESC LIMIT 1",
                        (event["run_id"],),
                    )
                    if done:
                        message = json.loads(done[0]["payload"]).get("final_message")
                        if message and (output := parts(message)):
                            task.artifacts.append(
                                proto.Artifact(
                                    artifact_id=f"{event['run_id']}-answer",
                                    name="answer",
                                    parts=output,
                                )
                            )
                payload = notification_payload(task, config["protocol"])
                if len(payload.encode()) > 24 * 1024 * 1024:
                    task.ClearField("artifacts")
                    task.metadata["artifacts_omitted"] = (
                        "Fetch the owned task or session export; callback exceeded 24 MiB"
                    )
                    payload = notification_payload(task, config["protocol"])
                queued.append(
                    (
                        delivery_id,
                        config["owner"],
                        config["task_id"],
                        config["id"],
                        event["seq"],
                        payload,
                        "pending",
                        0,
                        time.time(),
                        None,
                    )
                )
            if events:
                async with self.service.store.connection() as db:
                    await db.execute("BEGIN IMMEDIATE")
                    async with db.execute(
                        "SELECT active FROM api_a2a_callbacks WHERE owner=? AND task_id=? AND id=?",
                        (config["owner"], config["task_id"], config["id"]),
                    ) as cursor:
                        active = await cursor.fetchone()
                    if active and active[0]:
                        await db.executemany(
                            "INSERT OR IGNORE INTO api_a2a_callback_deliveries VALUES (?,?,?,?,?,?,?,?,?,?)",
                            queued,
                        )
                        await db.execute(
                            "UPDATE api_a2a_callbacks SET last_seq=MAX(last_seq,?) WHERE owner=? AND task_id=? AND id=?",
                            (events[-1]["seq"], config["owner"], config["task_id"], config["id"]),
                        )
                    await db.commit()

    async def tick(self, *, moment: float | None = None) -> None:
        async with self._lock:
            await self.collect()
            moment = time.time() if moment is None else moment
            rows = await self.service.store.rows(
                "SELECT d.*,c.config,c.protocol FROM api_a2a_callback_deliveries d JOIN api_a2a_callbacks c ON c.owner=d.owner AND c.task_id=d.task_id AND c.id=d.config_id WHERE d.state='pending' AND d.next_attempt<=? AND c.active=1 AND NOT EXISTS (SELECT 1 FROM api_a2a_callback_deliveries older WHERE older.owner=d.owner AND older.task_id=d.task_id AND older.config_id=d.config_id AND older.event_seq<d.event_seq AND older.state IN ('pending','sending')) ORDER BY d.event_seq,d.id LIMIT 20",
                (moment,),
            )
            for row in rows:
                if row["attempts"] >= self.max_attempts:
                    async with self.service.store.connection() as db:
                        await db.execute(
                            "UPDATE api_a2a_callback_deliveries SET state='failed',error='Notification attempts exhausted' WHERE id=? AND state='pending'",
                            (row["id"],),
                        )
                        await db.commit()
                    continue
                async with self.service.store.connection() as db:
                    changed = await db.execute(
                        "UPDATE api_a2a_callback_deliveries SET state='sending',attempts=attempts+1 WHERE id=? AND state='pending'",
                        (row["id"],),
                    )
                    await db.commit()
                if not changed.rowcount:
                    continue
                state, error, delay = "failed", "Notification attempts exhausted", 0.0
                config = ParseDict(json.loads(row["config"]), proto.TaskPushNotificationConfig())
                grant = self.grants.get((row["owner"], config.url))
                attempts = row["attempts"] + 1
                if grant and attempts <= self.max_attempts:
                    secret, bearer = grant
                    body = row["payload"].encode()
                    headers = {
                        "Content-Type": "application/json",
                        "A2A-Version": row["protocol"],
                        "X-A2A-Delivery-ID": row["id"],
                        "X-A2A-Signature": hmac.new(
                            secret.encode(), body, hashlib.sha256
                        ).hexdigest(),
                    }
                    if config.token:
                        headers["X-A2A-Notification-Token"] = config.token
                    if bearer:
                        headers["Authorization"] = "Bearer " + bearer
                    try:
                        code, retry_after = await self.http.post(config.url, body, headers)
                        if 200 <= code < 300:
                            state, error = "delivered", None
                        elif code == 429 or code >= 500:
                            state, error = "pending", f"Callback HTTP {code}"
                            with suppress(ValueError):
                                delay = max(0.0, min(float(retry_after), 3600))
                        else:
                            error = f"Callback HTTP {code}; redirects and permanent failures are not retried"
                    except InvalidParamsError:
                        error = "Callback destination failed public-address policy"
                    except (httpx.HTTPError, TimeoutError, OSError, ValueError):
                        state, error = (
                            "pending",
                            "Notification connection failed or delivery acknowledgement was lost",
                        )
                elif not grant:
                    error = "Callback endpoint grant was removed"
                if state == "pending" and attempts >= self.max_attempts:
                    state = "failed"
                next_attempt = moment + max(delay, min(300, 5 * 2 ** min(attempts - 1, 8)))
                async with self.service.store.connection() as db:
                    await db.execute(
                        "UPDATE api_a2a_callback_deliveries SET state=CASE WHEN ?='pending' AND NOT EXISTS (SELECT 1 FROM api_a2a_callbacks c WHERE c.owner=api_a2a_callback_deliveries.owner AND c.task_id=api_a2a_callback_deliveries.task_id AND c.id=api_a2a_callback_deliveries.config_id AND c.active=1) THEN 'cancelled' ELSE ? END,error=?,next_attempt=? WHERE id=?",
                        (state, state, error, next_attempt, row["id"]),
                    )
                    await db.commit()

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
                self.last_error = None
            except Exception:
                # A transient database/serialization failure retains durable
                # cursors and pending records. Never retry agent execution.
                self.last_error = (
                    "Callback processing failed; durable delivery records were retained"
                )
            await asyncio.sleep(0.5)


def callbacks(service: HarnessService) -> CallbackManager:
    if service.a2a_callbacks is None:
        raise PushNotificationNotSupportedError()
    return service.a2a_callbacks


def notification_payload(task: proto.Task, protocol: str) -> str:
    if protocol == "0.3":
        value = to_compat_task(task).model_dump(mode="json", by_alias=True, exclude_none=True)
    elif protocol == "1.0":
        value = MessageToDict(proto.StreamResponse(task=task))
    else:
        raise ValueError("Unsupported persisted callback protocol")
    # Match pinned Hermes canonical signing bytes. Sending these same bytes
    # also supports ordinary raw-body HMAC verification without reparsing JSON.
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
