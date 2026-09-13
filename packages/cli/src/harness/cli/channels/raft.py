"""Raft content-free wake bridge and explicitly scoped approved CLI tools."""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import web

from harness.cli.channels.process import client_environment, run_client, stop_client
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.tools import Tool

WAKE_FIELDS = frozenset(
    {
        "schema",
        "attemptId",
        "eventId",
        "messageId",
        "agentId",
        "profile",
        "coreSessionId",
        "adapterInstance",
        "occurredAt",
    }
)


async def raft_identity(config: ChannelConfig) -> dict[str, Any]:
    code, output = await run_client(
        [config.command or "raft", "--profile", config.username, "auth", "whoami"],
        env={"RAFT_PROFILE": config.username},
    )
    if code:
        raise ChannelError("Raft profile could not be loaded")
    payload = json.loads(output)
    identity = payload.get("data", {})
    if (
        not payload.get("ok")
        or identity.get("agentId") != config.app_id
        or identity.get("profileSlug") != config.username
        or str(identity.get("serverUrl", "")).rstrip("/") != config.homeserver.rstrip("/")
    ):
        raise ChannelError("Raft credentials do not match the configured profile, agent and server")
    return identity


class RaftTransport(WebhookTransport):
    name = "raft"
    limit = 4000

    async def authenticate(self) -> None:
        if (
            self.config.listen_host not in {"127.0.0.1", "::1", "localhost"}
            or not self.config.username
            or not self.config.app_id
            or len(self.token) < 32
        ):
            raise ChannelError(
                "Raft requires a loopback listener, explicit profile/agent and >=32-character bridge token"
            )
        self.bot_id = self.config.app_id
        self.identity = json.dumps([self.config.homeserver, self.config.username, self.bot_id])
        await raft_identity(self.config)
        self.store: ChannelStore | None = None

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        if not hmac.compare_digest(
            self.token.encode(), headers.get("X-Raft-Bridge-Token", "").encode()
        ):
            raise AuthenticationError("Raft bridge authentication failed")
        if len(body) > 16384:
            raise ValueError("Raft wake is too large")
        payload = json.loads(body)
        if (
            not isinstance(payload, dict)
            or set(payload) - WAKE_FIELDS
            or any(
                value is not None and (not isinstance(value, str) or len(value) > 256)
                for value in payload.values()
            )
        ):
            raise ValueError("Raft wake must contain only bounded protocol metadata")
        if (
            payload.get("schema") != "raft-channel-wake.v1"
            or payload.get("profile") != self.config.username
            or payload.get("agentId") != self.bot_id
        ):
            raise AuthenticationError("Raft wake profile/agent mismatch")
        identifier = payload.get("eventId") or payload.get("attemptId")
        if not identifier:
            raise ValueError("Raft wake requires a stable delivery identity")
        self.accept(
            ChannelMessage(
                id=identifier,
                user_id="raft:" + self.config.username,
                channel_id=self.config.username,
                thread_id=self.config.username,
                text="A Raft wake notice arrived. Use raft_manual to inspect the operating guide when needed, then the scoped Raft tools to read and handle pending work. Inbox drains and outbound messages require approval.",
            ),
            store,
        )
        self.store = store
        return {"ok": True, "runtimeSession": self.config.username}

    def application(self, store: ChannelStore) -> web.Application:
        app = super().application(store)

        async def drain(request: web.Request) -> web.Response:
            if not hmac.compare_digest(
                self.token.encode(), request.headers.get("X-Raft-Bridge-Token", "").encode()
            ):
                return web.json_response({"error": "unauthorized"}, status=401)
            # Activity mirroring is not enabled; explicitly return an empty drain
            # contract so the official bridge need not probe a missing endpoint.
            return web.json_response(
                {"schema": "raft-activity-drain.v1", "events": [], "dropped": 0}
            )

        app.router.add_get("/activity/drain", drain)
        return app

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        runner = web.AppRunner(self.application(store), access_log=None, shutdown_timeout=5)
        await runner.setup()
        process = None
        try:
            await web.TCPSite(runner, self.config.listen_host, self.config.listen_port).start()
            host = "[::1]" if self.config.listen_host == "::1" else self.config.listen_host
            endpoint = f"http://{host}:{self.config.listen_port}{self.config.webhook_path}"
            process = await asyncio.create_subprocess_exec(
                self.config.command or "raft",
                "--profile",
                self.config.username,
                "agent",
                "bridge",
                "--wake-adapter",
                "wake-channel",
                "--wake-channel-endpoint",
                endpoint,
                env=client_environment(
                    {"RAFT_PROFILE": self.config.username, "RAFT_CHANNEL_TOKEN": self.token}
                ),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            store.set("connection", "bridge_running")
            await process.wait()
            raise ChannelError("Raft bridge exited; the channel runner will reconnect")
        finally:
            if process is not None:
                await stop_client(process)
            await runner.cleanup()

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        if thread_id != self.config.username or self.store is None:
            raise ChannelError("Raft local response belongs to a different profile")
        response = {"id": delivery_id, "destination": "local", "profile": thread_id, "text": text}
        self.store.set("local_response:" + delivery_id, response)
        responses = [
            item for item in self.store.get("local_responses", []) if item.get("id") != delivery_id
        ]
        self.store.set("local_responses", [*responses, response][-30:])


class _RaftTool:
    def __init__(self, config: ChannelConfig, operation: str):
        self.config, self.operation = config, operation
        self.name = "raft_" + operation
        self.description = {
            "manual": "Read the configured Raft profile's operating guide; explain intent and reason.",
            "check": "Drain and acknowledge the configured Raft profile's pending inbox. Requires approval because it consumes messages.",
            "read": "Read bounded history for an allowed Raft target. Records read context in the configured CLI profile.",
            "send": "Send the exact reviewed text to one configured Raft target. Uncertain outcomes must not be automatically retried.",
        }[operation]
        self.approval: ApprovalDecision = "auto" if operation == "manual" else "prompt"
        self.effect_scope = "read_only" if operation == "manual" else "external_side_effect"
        properties = {
            "manual": {
                "topic": {"type": "string"},
                "intent": {"type": "string", "minLength": 12, "maxLength": 500},
                "reason": {"type": "string", "minLength": 12, "maxLength": 500},
            },
            "check": {},
            "read": {"target": {"type": "string"}},
            "send": {"target": {"type": "string"}, "text": {"type": "string", "maxLength": 16000}},
        }[operation]
        self.parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            await raft_identity(self.config)
            args = call.arguments
            text = None
            if self.operation == "manual":
                topic = str(args["topic"])
                if (
                    not topic
                    or len(topic) > 100
                    or not all(char.isalnum() or char in "-_" for char in topic)
                ):
                    raise ChannelError("Invalid Raft manual topic")
                command = [
                    "manual",
                    "get",
                    topic,
                    "--intent",
                    str(args["intent"]),
                    "--reason",
                    str(args["reason"]),
                ]
            elif self.operation == "check":
                command = ["message", "check"]
            else:
                target = str(args["target"])
                if (
                    not target
                    or len(target) > 256
                    or any(char.isspace() for char in target)
                    or not any(
                        target == allowed or target.startswith(allowed + ":")
                        for allowed in self.config.allowed_targets
                    )
                ):
                    raise ChannelError("Raft target is outside the configured allowlist")
                command = ["message", self.operation, "--target=" + target]
                if self.operation == "send":
                    text = str(args["text"])
                    if len(text) > 16000:
                        raise ChannelError("Raft outgoing text exceeds its bound")
                    command += ["--json"]
                else:
                    command += ["--limit", "100"]
            code, output = await run_client(
                [self.config.command or "raft", "--profile", self.config.username, *command],
                env={"RAFT_PROFILE": self.config.username},
                input_text=text,
            )
            if code:
                raise ChannelError("Raft operation failed; a mutation outcome may be uncertain")
            return ToolResult(tool_call_id=call.id, name=self.name, content=output)
        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="Raft operation failed or was interrupted. Inspect the configured profile before retrying a mutation.",
                is_error=True,
            )


def raft_tools(config: ChannelConfig, *, user_id: str, cwd: Path) -> list[Tool]:
    owner = "raft:" + config.username
    if user_id != owner or owner not in config.allowed_users or not config.app_id:
        raise ChannelError("Raft tool ownership does not match its explicitly configured channel")
    return [_RaftTool(config, operation) for operation in ("manual", "check", "read", "send")]
