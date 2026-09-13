"""Configured remote-agent peers with scoped history and durable send claims."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.cli.account_auth import validated_endpoint
from harness.core import Session, Tool, ToolCall, ToolResult
from harness.core.memory import MemoryScope
from harness.core.paths import user_home
from harness.core.schemas import ApprovalDecision, EffectScope
from harness.core.secret_redaction import redact_secrets
from harness.core.session_search import session_scope


class A2APeer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    url: str
    token_env: str = ""
    capabilities: tuple[str, ...] = ()
    timeout: float = Field(default=120, gt=0, le=600)
    protocol: Literal["1.0", "0.3"] = "1.0"

    @model_validator(mode="after")
    def endpoint(self):
        validated_endpoint(self.url)
        return self


class A2AConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    peers: dict[str, A2APeer] = Field(default_factory=dict)
    max_parallel: int = Field(default=6, ge=1, le=6)
    max_context_turns: int = Field(default=20, ge=1, le=100)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1024, le=8 * 1024 * 1024)
    poll_interval: float = Field(default=0.5, ge=0.05, le=10)

    @model_validator(mode="after")
    def names(self):
        if len(self.peers) > 100 or any(
            not name or len(name) > 64 or not name.replace("-", "").replace("_", "").isalnum()
            for name in self.peers
        ):
            raise ValueError("Configure at most 100 peers with simple names")
        return self


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PeerHTTPError(ValueError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Peer request failed (HTTP {status_code})")


class PeerHistory:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("Peer history must not be a symlink")
        fd = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        os.close(fd)
        if self.path.is_symlink():
            raise ValueError("Peer history must not be a symlink")
        os.chmod(self.path, 0o600)
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS peer_contexts(id TEXT PRIMARY KEY,scope TEXT NOT NULL,peer TEXT NOT NULL,peer_key TEXT NOT NULL,remote_context TEXT,remote_task TEXT,state TEXT NOT NULL,updated REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS peer_exchanges(id TEXT PRIMARY KEY,context_id TEXT NOT NULL,scope TEXT NOT NULL,operation TEXT NOT NULL,peer_key TEXT NOT NULL,prompt TEXT NOT NULL,reply TEXT NOT NULL,state TEXT NOT NULL,created REAL NOT NULL,UNIQUE(scope,operation,peer_key))"
            )

    def contexts(self, scope):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,peer,state,updated FROM peer_contexts WHERE scope=? ORDER BY updated DESC LIMIT 100",
                    (scope,),
                )
            ]

    def context(self, scope, identifier):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM peer_contexts WHERE scope=? AND id=?", (scope, identifier)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown owned peer conversation")
            return dict(row)

    def history(self, scope, identifier):
        self.context(scope, identifier)
        with self.connection() as db:
            rows = list(
                db.execute(
                    "SELECT id,prompt,reply,state,created FROM peer_exchanges WHERE scope=? AND context_id=? ORDER BY created DESC LIMIT 50",
                    (scope, identifier),
                )
            )
            return [dict(row) for row in reversed(rows)]

    def claim(self, scope, peer, peer_key, operation, prompt, identifier, max_turns):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM peer_exchanges WHERE scope=? AND operation=? AND peer_key=?",
                (scope, operation, peer_key),
            ).fetchone():
                raise ValueError(
                    "This tool call already submitted or has an uncertain outcome; inspect peer history"
                )
            if identifier:
                found = db.execute(
                    "SELECT * FROM peer_contexts WHERE id=? AND scope=? AND peer_key=?",
                    (identifier, scope, peer_key),
                ).fetchone()
                if found is None:
                    raise ValueError("Unknown conversation for this peer and account")
                context = dict(found)
                if context["state"] in {
                    "sending",
                    "queued",
                    "working",
                    "uncertain",
                    "cancel_requested",
                }:
                    raise ValueError("The prior peer request is unfinished; inspect a2a_status")
                if not context["remote_context"]:
                    raise ValueError("The peer did not provide a resumable context")
                count = db.execute(
                    "SELECT COUNT(*) FROM peer_exchanges WHERE context_id=?", (identifier,)
                ).fetchone()[0]
                if count >= max_turns:
                    raise ValueError(
                        "Peer conversation turn limit reached; start a new explicit context"
                    )
            else:
                identifier = "a2actx_" + uuid4().hex
                context = {
                    "id": identifier,
                    "remote_context": None,
                    "remote_task": None,
                    "state": "new",
                }
                db.execute(
                    "INSERT INTO peer_contexts VALUES (?,?,?,?,?,?,?,?)",
                    (identifier, scope, peer, peer_key, None, None, "new", time.time()),
                )
            message_id = "a2amsg_" + uuid4().hex
            db.execute(
                "INSERT INTO peer_exchanges VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    message_id,
                    identifier,
                    scope,
                    operation,
                    peer_key,
                    prompt,
                    "",
                    "sending",
                    time.time(),
                ),
            )
            db.execute(
                "UPDATE peer_contexts SET state='sending',remote_task=?,updated=? WHERE id=?",
                (
                    context["remote_task"] if context["state"] == "input_required" else None,
                    time.time(),
                    identifier,
                ),
            )
            return context, message_id

    def finish(
        self, scope, identifier, message_id, state, reply="", remote_context=None, remote_task=None
    ):
        with self.connection() as db:
            db.execute(
                "UPDATE peer_contexts SET state=?,updated=?,remote_context=COALESCE(?,remote_context),remote_task=COALESCE(?,remote_task) WHERE id=? AND scope=?",
                (state, time.time(), remote_context, remote_task, identifier, scope),
            )
            if message_id:
                db.execute(
                    "UPDATE peer_exchanges SET state=?,reply=? WHERE id=? AND scope=?",
                    (state, reply, message_id, scope),
                )
            else:
                db.execute(
                    "UPDATE peer_exchanges SET state=?,reply=? WHERE id=(SELECT id FROM peer_exchanges WHERE context_id=? AND scope=? ORDER BY created DESC LIMIT 1)",
                    (state, reply, identifier, scope),
                )


class A2AToolset:
    def __init__(
        self,
        config: A2AConfig,
        *,
        home: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config, self.home, self.transport = config, home or user_home(), transport
        self.store = PeerHistory(self.home / "integrations/a2a.sqlite3")
        self.client: httpx.AsyncClient | None = None
        self.tokens: dict[str, str] = {}
        self.peer_keys: dict[str, str] = {}
        self.semaphore = asyncio.Semaphore(config.max_parallel)

    async def __aenter__(self):
        if not self.config.enabled:
            return self
        for name, peer in self.config.peers.items():
            token = os.environ.get(peer.token_env, "") if peer.token_env else ""
            if peer.token_env and not token:
                raise ValueError(f"Missing A2A credential variable {peer.token_env}")
            self.tokens[name] = token
            self.peer_keys[name] = _hash([name, peer.model_dump(), token])
        await asyncio.to_thread(self.store.initialize)
        self.client = httpx.AsyncClient(
            transport=self.transport, follow_redirects=False, trust_env=False
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
            self.client = None
        self.tokens.clear()

    def bind(self, session: Session) -> list[Tool]:
        if not self.config.enabled:
            return []
        scope = session_scope(session)
        if scope is None:
            if "memory_scope" in session.metadata:
                return []
            scope = MemoryScope(workspace=str(session.cwd.resolve()))
        key = _hash([str(self.home.resolve()), scope.model_dump()])
        return [
            PeerTool(self, key, session.id, name)
            for name in (
                "a2a_list",
                "a2a_history",
                "a2a_discover",
                "a2a_call",
                "a2a_status",
                "a2a_cancel",
                "a2a_orchestrate",
            )
        ]

    def sanitize(self, value: str) -> str:
        for token in self.tokens.values():
            if token:
                value = value.replace(token, "[REDACTED:peer-token]")
        return redact_secrets(value)[0]

    async def request(self, name, method, url, body=None):
        assert self.client is not None
        peer = self.config.peers[name]
        expected, actual = httpx.URL(peer.url), httpx.URL(url)
        if (
            (expected.scheme, expected.host, expected.port)
            != (actual.scheme, actual.host, actual.port)
            or actual.username
            or actual.password
            or actual.fragment
        ):
            raise ValueError("Peer advertised an endpoint outside its configured origin")
        headers = {"A2A-Version": peer.protocol}
        if self.tokens[name]:
            headers["Authorization"] = "Bearer " + self.tokens[name]
        async with self.client.stream(
            method, url, json=body, headers=headers, timeout=peer.timeout
        ) as response:
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > self.config.max_response_bytes:
                    raise ValueError("Peer response exceeded its size limit")
            if response.status_code != 200:
                raise PeerHTTPError(response.status_code)
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("Peer response must be an object")
            return payload

    async def discover(self, name):
        peer = self.config.peers.get(name)
        if peer is None:
            raise ValueError("Choose a configured peer from a2a_list")
        base = httpx.URL(peer.url)
        try:
            return await self.request(
                name, "GET", str(base.copy_with(path="/.well-known/agent-card.json", query=None))
            )
        except PeerHTTPError as exc:
            if exc.status_code != 404:
                raise
            return await self.request(
                name, "GET", str(base.copy_with(path="/.well-known/agent.json", query=None))
            )

    async def endpoint(self, name):
        card = await self.discover(name)
        peer = self.config.peers[name]
        for interface in card.get("supportedInterfaces", []) or []:
            if (
                isinstance(interface, dict)
                and interface.get("protocolBinding") == "JSONRPC"
                and str(interface.get("protocolVersion", "1.0")) == peer.protocol
            ):
                return str(interface["url"])
        return str(card.get("url") or peer.url)

    async def rpc(self, name, endpoint, method, params):
        if self.config.peers[name].protocol == "0.3":
            method = {
                "SendMessage": "message/send",
                "GetTask": "tasks/get",
                "CancelTask": "tasks/cancel",
            }[method]
        request_id = uuid4().hex
        payload = await self.request(
            name,
            "POST",
            endpoint,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        if payload.get("id") != request_id or payload.get("jsonrpc") != "2.0":
            raise ValueError("Peer returned a mismatched JSON-RPC response")
        if "error" in payload:
            raise ValueError("Peer rejected the request; inspect its permissions and task status")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("Peer returned an invalid task or message")
        return result.get("task") or result.get("message") or result

    def outcome(self, result):
        if not isinstance(result, dict) or not (
            ("status" in result and isinstance(result["status"], dict) and result.get("id"))
            or (isinstance(result.get("parts"), list) and result["parts"])
        ):
            raise ValueError("Peer returned neither a task nor a message")
        if "status" in result and not isinstance(result["status"], dict):
            raise ValueError("Peer task status must be an object")
        status = result.get("status") or {}
        if "status" in result and not isinstance(status.get("state"), str):
            raise ValueError("Peer task has no valid state")
        state = (
            str(status.get("state") or "completed")
            .removeprefix("TASK_STATE_")
            .lower()
            .replace("-", "_")
        )
        state = {
            "submitted": "queued",
            "auth_required": "input_required",
            "rejected": "failed",
            "canceled": "cancelled",
        }.get(state, state)
        if state not in {"queued", "working", "input_required", "completed", "cancelled", "failed"}:
            raise ValueError("Peer returned an unknown task state")
        parts = []
        artifacts = result.get("artifacts", []) or []
        if not isinstance(artifacts, list):
            raise ValueError("Peer artifacts must be a list")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("parts", []), list):
                raise ValueError("Peer artifact has invalid parts")
            parts.extend(artifact.get("parts", []))
        if not parts:
            message = status.get("message") or result
            if not isinstance(message, dict) or not isinstance(message.get("parts", []), list):
                raise ValueError("Peer message has invalid parts")
            parts = message.get("parts", [])
        text = self.sanitize(
            "\n".join(
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        )[:64000]
        context, task = result.get("contextId"), result.get("id") if "status" in result else None
        if any(
            value is not None and (not isinstance(value, str) or not value or len(value) > 1024)
            for value in (context, task)
        ):
            raise ValueError("Peer returned invalid context/task identifiers")
        return state, text, context, task

    async def save_result(self, scope, context, message_id, result):
        state, text, remote_context, task = self.outcome(result)
        await asyncio.to_thread(
            self.store.finish, scope, context, message_id, state, text, remote_context, task
        )
        return {"context_id": context, "state": state, "text": text, "task_id": task}

    async def call(self, scope, operation, name, message, context_id=None):
        if (
            name not in self.config.peers
            or not isinstance(message, str)
            or not message.strip()
            or len(message) > 16000
        ):
            raise ValueError(
                "Choose a configured peer and a nonempty message of at most 16000 characters"
            )
        async with self.semaphore:
            endpoint = await self.endpoint(name)
            context, message_id = await asyncio.to_thread(
                self.store.claim,
                scope,
                name,
                self.peer_keys[name],
                operation,
                self.sanitize(message),
                context_id,
                self.config.max_context_turns,
            )
            identifier = context["id"]
            message = {
                "messageId": message_id,
                "role": "ROLE_USER",
                "parts": [{"text": self.sanitize(message)}],
            }
            if context["remote_context"]:
                message["contextId"] = context["remote_context"]
            if context["state"] == "input_required" and context["remote_task"]:
                message["taskId"] = context["remote_task"]
            configuration = {"returnImmediately": True}
            if self.config.peers[name].protocol == "0.3":
                message["role"] = "user"
                message["parts"][0]["kind"] = "text"
                configuration = {"blocking": False}
            sent = None
            try:
                async with asyncio.timeout(self.config.peers[name].timeout):
                    result = await self.rpc(
                        name,
                        endpoint,
                        "SendMessage",
                        {"message": message, "configuration": configuration},
                    )
                    sent = await self.save_result(scope, identifier, message_id, result)
                    while sent["state"] in {"queued", "working"} and sent["task_id"]:
                        await asyncio.sleep(self.config.poll_interval)
                        result = await self.rpc(name, endpoint, "GetTask", {"id": sent["task_id"]})
                        sent = await self.save_result(scope, identifier, message_id, result)
                    return {"peer": name, **sent}
            except TimeoutError:
                if sent:
                    return {
                        "peer": name,
                        **sent,
                        "notice": "Peer is still active; inspect a2a_status. No request was repeated.",
                    }
                await asyncio.to_thread(
                    self.store.finish, scope, identifier, message_id, "uncertain"
                )
                return {
                    "peer": name,
                    "context_id": identifier,
                    "state": "uncertain",
                    "notice": "Submission response was lost. Inspect the peer before starting another request.",
                }
            except BaseException:
                if sent and sent["task_id"]:
                    # The durable handle survives cancellation. No implicit
                    # replay or claim that remote work has been undone.
                    pass
                else:
                    await asyncio.to_thread(
                        self.store.finish, scope, identifier, message_id, "uncertain"
                    )
                raise

    async def status(self, scope, identifier, cancel=False):
        context = await asyncio.to_thread(self.store.context, scope, identifier)
        name = context["peer"]
        if name not in self.peer_keys or context["peer_key"] != self.peer_keys[name]:
            raise ValueError("This peer account changed; inspect its saved history locally")
        if not context["remote_task"]:
            return {
                "context_id": identifier,
                "state": context["state"],
                "notice": "No remote task handle was returned; do not assume an uncertain send failed.",
            }
        endpoint = await self.endpoint(name)
        result = await self.rpc(
            name, endpoint, "CancelTask" if cancel else "GetTask", {"id": context["remote_task"]}
        )
        return await self.save_result(scope, identifier, None, result)


class PeerTool:
    def __init__(self, owner: A2AToolset, scope: str, session_id: str, name: str):
        self.owner, self.scope, self.session_id, self.name = owner, scope, session_id, name
        self.approval: ApprovalDecision = (
            "prompt" if name in {"a2a_call", "a2a_orchestrate", "a2a_cancel"} else "auto"
        )
        self.effect_scope: EffectScope = (
            "external_side_effect" if self.approval == "prompt" else "read_only"
        )
        self.description = {
            "a2a_list": "List configured remote-agent peers and owned saved conversations.",
            "a2a_history": "Read the saved transcript of an owned peer conversation.",
            "a2a_discover": "Inspect the Agent Card of one explicitly configured peer.",
            "a2a_call": "Ask a configured peer to perform work; optionally continue an owned context.",
            "a2a_status": "Refresh the status of an owned remote task without repeating its submission.",
            "a2a_cancel": "Request cancellation of an owned remote task; completed effects may remain.",
            "a2a_orchestrate": "Ask peers with a configured capability in parallel. all returns every result, first returns the first successful completion, best chooses the longest successful text (not a quality score).",
        }[name]
        string = {"type": "string", "minLength": 1, "maxLength": 16000}
        properties = {
            "a2a_list": {},
            "a2a_history": {"context_id": string},
            "a2a_discover": {"peer": string},
            "a2a_call": {"peer": string, "message": string, "context_id": string},
            "a2a_status": {"context_id": string},
            "a2a_cancel": {"context_id": string},
            "a2a_orchestrate": {
                "capability": string,
                "message": string,
                "mode": {"type": "string", "enum": ["all", "first", "best"]},
            },
        }[name]
        required = [
            key
            for key in properties
            if key not in {"mode", "context_id"} or (name != "a2a_call" and key == "context_id")
        ]
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        try:
            if self.name == "a2a_list":
                result = {
                    "peers": [
                        {
                            "name": name,
                            "capabilities": list(peer.capabilities),
                            "protocol": peer.protocol,
                        }
                        for name, peer in self.owner.config.peers.items()
                    ],
                    "conversations": await asyncio.to_thread(self.owner.store.contexts, self.scope),
                }
            elif self.name == "a2a_history":
                result = await asyncio.to_thread(
                    self.owner.store.history, self.scope, args["context_id"]
                )
            elif self.name == "a2a_discover":
                result = await self.owner.discover(args["peer"])
            elif self.name == "a2a_call":
                result = await self.owner.call(
                    self.scope,
                    self.session_id + ":" + call.id,
                    args["peer"],
                    args["message"],
                    args.get("context_id"),
                )
            elif self.name in {"a2a_status", "a2a_cancel"}:
                result = await self.owner.status(
                    self.scope, args["context_id"], cancel=self.name == "a2a_cancel"
                )
            else:
                peers = [
                    name
                    for name, peer in self.owner.config.peers.items()
                    if args["capability"] == "*" or args["capability"] in peer.capabilities
                ]
                if not peers:
                    raise ValueError("No configured peer advertises that capability")
                mode = args.get("mode", "all")
                if mode not in {"all", "first", "best"}:
                    raise ValueError("Unknown orchestration mode")

                async def run(name):
                    try:
                        return await self.owner.call(
                            self.scope, self.session_id + ":" + call.id, name, args["message"]
                        )
                    except Exception:
                        return {
                            "peer": name,
                            "state": "failed",
                            "text": "Peer request failed; inspect owned history.",
                        }

                tasks = [asyncio.create_task(run(name)) for name in peers]
                results = []
                try:
                    for task in asyncio.as_completed(tasks):
                        item = await task
                        results.append(item)
                        if mode == "first" and item["state"] == "completed":
                            break
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                successes = [item for item in results if item["state"] == "completed"]
                selected = (
                    max(successes, key=lambda item: len(item.get("text", "")))
                    if mode == "best" and successes
                    else results
                )
                result = {"mode": mode, "results": selected}
                if mode == "first":
                    result["notice"] = (
                        "Already submitted peers may still run; inspect a2a_list and cancel owned tasks explicitly."
                    )
            output = self.owner.sanitize(json.dumps(result, ensure_ascii=False))
            if len(output) > 64000:
                output = output[:64000] + "\n[truncated; use a narrower history/status query]"
            failed = isinstance(result, dict) and result.get("state") in {
                "failed",
                "cancelled",
                "uncertain",
            }
            return ToolResult(tool_call_id=call.id, name=self.name, content=output, is_error=failed)
        except (ValueError, TypeError, KeyError, httpx.HTTPError, sqlite3.Error) as exc:
            content = (
                str(exc)
                if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
                else "Peer operation failed; inspect configured endpoints and owned history."
            )
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=self.owner.sanitize(content),
                is_error=True,
            )
