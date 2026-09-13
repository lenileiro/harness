"""Scoped Honcho v3 user modeling with explicit, durable transcript export."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.cli.account_auth import validated_endpoint
from harness.core import ApprovalDecision, ConfigurationError, Session, Tool, ToolCall, ToolResult
from harness.core.memory import MemoryScope
from harness.core.paths import user_home
from harness.core.session_search import message_references, session_scope


class HonchoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    base_url: str = "https://api.honcho.dev"
    api_key_env: str = "HONCHO_API_KEY"
    workspace_prefix: str = Field(default="harness", pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    timeout: float = Field(default=60, gt=0, le=300)
    max_export_bytes: int = Field(default=256000, ge=1024, le=1024000)

    @model_validator(mode="after")
    def endpoint(self):
        validated_endpoint(self.base_url)
        return self


class HonchoToolset:
    def __init__(
        self,
        config: HonchoConfig,
        *,
        home: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config, self.home, self.transport = config, (home or user_home()).resolve(), transport
        self.client: httpx.AsyncClient | None = None
        self.database = self.home / "integrations/honcho.sqlite3"
        self.initialized: set[str] = set()
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ConfigurationError(f"Missing Honcho credential {self.config.api_key_env}")
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.config.timeout,
            transport=self.transport,
            trust_env=False,
            follow_redirects=False,
        )
        await asyncio.to_thread(self._initialize_ledger)
        return self

    async def __aexit__(self, *args):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        self.initialized.clear()

    def _initialize_ledger(self):
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with sqlite3.connect(self.database) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS exports (workspace TEXT NOT NULL, reference TEXT NOT NULL, state TEXT NOT NULL, PRIMARY KEY(workspace,reference))"
            )
        self.database.chmod(0o600)

    def _states(self, workspace: str) -> dict[str, str]:
        with sqlite3.connect(self.database) as db:
            return dict(
                db.execute("SELECT reference,state FROM exports WHERE workspace=?", (workspace,))
            )

    def _claim(self, workspace: str, references: list[str]) -> None:
        with sqlite3.connect(self.database, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            for reference in references:
                row = db.execute(
                    "SELECT state FROM exports WHERE workspace=? AND reference=?",
                    (workspace, reference),
                ).fetchone()
                if row is not None:
                    raise ValueError(
                        "A selected message is already exported or has an uncertain prior export; inspect honcho_status"
                    )
            db.executemany(
                "INSERT INTO exports VALUES (?,?,?)",
                [(workspace, ref, "uncertain") for ref in references],
            )

    def _complete(self, workspace: str, references: list[str]) -> None:
        with sqlite3.connect(self.database) as db:
            db.executemany(
                'UPDATE exports SET state="exported" WHERE workspace=? AND reference=?',
                [(workspace, ref) for ref in references],
            )

    def bind(self, session: Session) -> list[Tool]:
        scope = session_scope(session)
        if scope is None:
            if "memory_scope" in session.metadata:
                return []
            scope = MemoryScope(workspace=str(session.cwd))
        # Remote callers do not inherit a local profile's user-modeling account.
        if scope.user_id is not None:
            return []
        identity = json.dumps(
            [str(self.home), self.config.base_url, scope.model_dump()], sort_keys=True
        )
        workspace = (
            self.config.workspace_prefix + "-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        )
        return [
            HonchoTool(self, session, workspace, name)
            for name in ("honcho_status", "honcho_sync", "honcho_chat")
        ]

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        if self.client is None:
            raise RuntimeError("Honcho client is closed")
        async with self.client.stream("POST", path, json=payload) as response:
            if not response.is_success:
                raise ValueError(
                    f"Honcho returned HTTP {response.status_code}; request was not retried"
                )
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > self.config.max_export_bytes * 2:
                    raise ValueError("Honcho response exceeds the configured size limit")
            return json.loads(raw)

    async def ensure_workspace(self, workspace: str):
        if workspace in self.initialized:
            return
        await self.post("/v3/workspaces", {"id": workspace})
        for name, observe in [("user", True), ("assistant", False)]:
            await self.post(
                f"/v3/workspaces/{workspace}/peers",
                {"id": name, "configuration": {"observe_me": observe}},
            )
        self.initialized.add(workspace)


class HonchoTool:
    def __init__(self, owner: HonchoToolset, session: Session, workspace: str, name: str):
        self.owner, self.session, self.workspace, self.name = owner, session, workspace, name
        self.approval: ApprovalDecision = "auto" if name == "honcho_status" else "prompt"
        self.effect_scope: Literal["read_only", "external_side_effect"] = (
            "read_only" if name == "honcho_status" else "external_side_effect"
        )
        self.description = {
            "honcho_status": "Inspect locally tracked transcript export references. No network request.",
            "honcho_sync": "Send selected user/assistant messages from this session to the configured Honcho account for user modeling. Use references from honcho_status; system instructions, tools, and attachments are excluded.",
            "honcho_chat": "Ask the configured Honcho account about this local identity using its exported conversations. The query is sent to Honcho and may incur service charges.",
        }[name]
        properties = (
            {
                "references": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                }
            }
            if name == "honcho_sync"
            else {"query": {"type": "string", "minLength": 1, "maxLength": 4000}}
            if name == "honcho_chat"
            else {}
        )
        self.parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def messages(self):
        references = message_references(self.session.id, self.session.messages)
        return {
            ref: message
            for ref, message in zip(references, self.session.messages, strict=True)
            if message.role in {"user", "assistant"} and message.content
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            if self.name == "honcho_status":
                states = await asyncio.to_thread(self.owner._states, self.workspace)
                result = {
                    "workspace": self.workspace,
                    "messages": [
                        {
                            "reference": ref,
                            "role": message.role,
                            "preview": (message.content or "")[:160],
                            "state": states.get(ref, "not_exported"),
                        }
                        for ref, message in self.messages().items()
                    ][-100:],
                }
            else:
                async with self.owner.lock:
                    if self.name == "honcho_chat":
                        query = call.arguments.get("query")
                        if not isinstance(query, str) or not 0 < len(query) <= 4000:
                            raise ValueError("query must contain 1 to 4000 characters")
                        await self.owner.ensure_workspace(self.workspace)
                        result = await self.owner.post(
                            f"/v3/workspaces/{self.workspace}/peers/user/chat",
                            {"query": query, "stream": False},
                        )
                    else:
                        references = call.arguments.get("references")
                        if (
                            not isinstance(references, list)
                            or not 0 < len(references) <= 100
                            or not all(isinstance(ref, str) for ref in references)
                            or len(set(references)) != len(references)
                        ):
                            raise ValueError(
                                "references must contain 1 to 100 distinct message references"
                            )
                        messages = self.messages()
                        if any(ref not in messages for ref in references):
                            raise ValueError(
                                "A reference is missing or belongs to another conversation"
                            )
                        payload = [
                            {
                                "peer_id": messages[ref].role
                                if messages[ref].role == "user"
                                else "assistant",
                                "content": messages[ref].content,
                                "metadata": {"harness_reference": ref},
                            }
                            for ref in references
                        ]
                        if len(json.dumps(payload).encode()) > self.owner.config.max_export_bytes:
                            raise ValueError("Selected messages exceed the configured export limit")
                        await self.owner.ensure_workspace(self.workspace)
                        remote_session = (
                            "session-" + hashlib.sha256(self.session.id.encode()).hexdigest()[:32]
                        )
                        await self.owner.post(
                            f"/v3/workspaces/{self.workspace}/sessions",
                            {"id": remote_session, "peers": {"user": {}, "assistant": {}}},
                        )
                        # Persist uncertainty before the non-idempotent POST. Crashes,
                        # cancellation and lost responses never replay it automatically.
                        await asyncio.to_thread(self.owner._claim, self.workspace, references)
                        await self.owner.post(
                            f"/v3/workspaces/{self.workspace}/sessions/{remote_session}/messages",
                            {"messages": payload},
                        )
                        await asyncio.to_thread(self.owner._complete, self.workspace, references)
                        result = {
                            "exported": len(references),
                            "workspace": self.workspace,
                            "session": remote_session,
                        }
            return ToolResult(
                tool_call_id=call.id, name=call.name, content=json.dumps(result, ensure_ascii=False)
            )
        except (ValueError, OSError, RuntimeError, httpx.HTTPError, sqlite3.Error) as exc:
            detail = (
                str(exc)
                if isinstance(exc, ValueError) and not isinstance(exc, httpx.HTTPError)
                else f"Honcho operation failed ({type(exc).__name__}); inspect local export state before retrying"
            )
            return ToolResult(tool_call_id=call.id, name=call.name, content=detail, is_error=True)
