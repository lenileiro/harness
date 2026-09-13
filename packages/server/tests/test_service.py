from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from harness.core import Agent
from harness.core.events import Done, Event, TextDelta
from harness.core.failover import FailoverPolicy
from harness.core.schemas import ApprovalDecision, Capabilities, Message, ToolCall, ToolResult
from harness.core.tools import ToolRegistry
from harness.server import HarnessService, RunContext, RunSubmission, ServerAuth, create_app


class FakeAdapter:
    name = "fake"

    def __init__(self, *, block: asyncio.Event | None = None):
        self.calls = 0
        self.block = block
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id: str):
        self.cancelled.set()

    async def stream(
        self, *, model: str, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[Event]:
        self.calls += 1
        self.entered.set()
        try:
            if self.block:
                await self.block.wait()
            yield TextDelta(text="Hello")
            yield Done(final_message=Message(role="assistant", content="Hello from the test model"))
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class Arguments(BaseModel):
    text: str


class RecordingTool:
    name = "record"
    description = "Record a value"
    parameters_schema = Arguments.model_json_schema()
    approval: ApprovalDecision = "auto"
    effect_scope = "workspace_durable"
    phases = ()

    def __init__(self):
        self.values = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.values.append(call.arguments["text"])
        return ToolResult(tool_call_id=call.id, name=self.name, content="recorded")


def builder(adapter: FakeAdapter, tool: RecordingTool | None = None):
    def build(context: RunContext):
        tools = ToolRegistry()
        if tool:
            tools.register(tool)
        return Agent(
            adapters={"fake": adapter},
            tools=tools,
            storage=context.storage,
            failover=FailoverPolicy(chain=["fake"], max_attempts=1),
            default_model="test",
            memory_scope=None,
        )

    return build


@asynccontextmanager
async def client_for(service: HarnessService, monkeypatch: pytest.MonkeyPatch, **kwargs):
    monkeypatch.setenv("TEST_ALICE_TOKEN", "alice-token-" + "a" * 40)
    monkeypatch.setenv("TEST_BOB_TOKEN", "bob-token-" + "b" * 40)
    auth = ServerAuth(token_envs={"alice": "TEST_ALICE_TOKEN", "bob": "TEST_BOB_TOKEN"})
    app = create_app(service, auth=auth, **kwargs)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer alice-token-" + "a" * 40},
        ) as client,
    ):
        yield client


def bob_headers():
    return {"Authorization": "Bearer bob-token-" + "b" * 40}


async def terminal(client: httpx.AsyncClient, run_id: str):
    async with asyncio.timeout(5):
        while True:
            response = await client.get(f"/v1/runs/{run_id}")
            assert response.status_code == 200, response.text
            run = response.json()
            if run["state"] not in ("queued", "running"):
                return run
            await asyncio.sleep(0.005)


async def test_http_run_events_scope_search_and_versioned_export(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    async with client_for(service, monkeypatch) as client:
        unauthenticated = await client.get("/v1/health", headers={"Authorization": ""})
        assert unauthenticated.status_code == 401
        for extra in (
            {"Origin": "https://evil.example"},
            {"Host": "evil.example"},
            {"Origin": "null"},
        ):
            assert (await client.get("/v1/health", headers=extra)).status_code in (400, 403)
        response = await client.post("/v1/runs", json={"prompt": "Searchable unique project fact"})
        assert response.status_code == 202, response.text
        run = await terminal(client, response.json()["id"])
        assert run["state"] == "completed", run
        session_id = run["session_id"]
        events = await client.get(f"/v1/runs/{run['id']}/events")
        assert events.status_code == 200
        payloads = [
            json.loads(line[6:]) for line in events.text.splitlines() if line.startswith("data: ")
        ]
        assert payloads[-1]["state"] == "completed"
        assert any(event["type"] == "text_delta" for event in payloads)
        assert all("messages" not in event for event in payloads)
        ids = [int(line[4:]) for line in events.text.splitlines() if line.startswith("id: ")]
        replay = await client.get(
            f"/v1/runs/{run['id']}/events", headers={"Last-Event-ID": str(ids[-2])}
        )
        assert replay.text.count("data: ") == 1
        assert (await client.get("/v1/sessions?q=Searchable")).json()["sessions"][0][
            "session_id"
        ] == session_id
        for path in (
            f"/v1/runs/{run['id']}",
            f"/v1/runs/{run['id']}/events",
            f"/v1/sessions/{session_id}/messages",
            f"/v1/sessions/{session_id}/export",
        ):
            assert (await client.get(path, headers=bob_headers())).status_code == 404
        assert (await client.get("/v1/sessions", headers=bob_headers())).json()["sessions"] == []
        assert (
            await client.post(
                "/v1/runs",
                headers=bob_headers(),
                json={"prompt": "steal", "session_id": session_id},
            )
        ).status_code == 404
        export = (await client.get(f"/v1/sessions/{session_id}/export")).json()
        assert export["format"] == "harness.trajectory" and export["version"] == 1
        assert export["session"]["metadata"]["memory_scope"]["user_id"] == "alice"
        assert [item["role"] for item in export["messages"]] == ["user", "assistant"]
        resumed = await client.post(f"/v1/runs/{run['id']}/resume", json={"prompt": "follow up"})
        assert resumed.status_code == 202
        assert resumed.json()["resumed_from"] == run["id"]
        assert (await terminal(client, resumed.json()["id"]))["state"] == "completed"


async def test_tool_rpc_queues_approval_and_replays_once_through_agent(tmp_path, monkeypatch):
    tool = RecordingTool()
    adapter = FakeAdapter()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(adapter, tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch) as client:
        response = await client.post(
            "/v1/tool-runs", json={"name": "record", "arguments": {"text": "one"}}
        )
        assert response.status_code == 202, response.text
        run = await terminal(client, response.json()["id"])
        assert run["state"] == "paused", run
        assert tool.values == []
        assert adapter.calls == 0
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        assert (await client.get("/v1/approvals", headers=bob_headers())).json()["approvals"] == []
        assert (
            await client.post(
                f"/v1/approvals/{approval['id']}/resolve",
                json={"granted": True},
                headers=bob_headers(),
            )
        ).status_code == 404
        assert (await client.post(f"/v1/runs/{run['id']}/resume", json={})).status_code == 409
        assert (
            await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        ).status_code == 200
        assert (
            await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": False})
        ).status_code == 409
        resumed = await client.post(f"/v1/runs/{run['id']}/resume", json={})
        assert resumed.status_code == 202, resumed.text
        assert (await terminal(client, resumed.json()["id"]))["state"] == "completed"
        assert tool.values == ["one"]
        export = (await client.get(f"/v1/sessions/{run['session_id']}/export")).json()
        calls = [
            call["id"] for message in export["messages"] for call in message.get("tool_calls") or []
        ]
        results = [
            message["tool_call_id"] for message in export["messages"] if message["role"] == "tool"
        ]
        assert calls == results and len(calls) == 1
        replay_again = await client.post(f"/v1/runs/{run['id']}/resume", json={})
        await terminal(client, replay_again.json()["id"])
        assert tool.values == ["one"]
        assert (await client.post("/v1/tool-runs", json={"name": "unknown"})).status_code == 403


async def test_cancel_is_owned_and_session_serialized(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter), max_workers=1)
    async with client_for(service, monkeypatch) as client:
        run = (await client.post("/v1/runs", json={"prompt": "wait"})).json()
        await asyncio.wait_for(adapter.entered.wait(), 2)
        assert (
            await client.post(
                "/v1/runs", json={"prompt": "conflict", "session_id": run["session_id"]}
            )
        ).status_code == 409
        assert (
            await client.post(f"/v1/runs/{run['id']}/cancel", headers=bob_headers())
        ).status_code == 404
        assert (await client.post(f"/v1/runs/{run['id']}/cancel")).json()["state"] == "cancelled"
        assert adapter.cancelled.is_set()
        assert (await terminal(client, run["id"]))["state"] == "cancelled"


async def test_restart_marks_inflight_interrupted_and_preserves_queued_batch(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    database = tmp_path / "api.db"
    service = HarnessService(database, tmp_path, builder(adapter), max_workers=1)
    async with client_for(service, monkeypatch) as client:
        response = await client.post(
            "/v1/batches",
            json={"runs": [{"prompt": "first"}, {"prompt": "second"}, {"prompt": "third"}]},
        )
        assert response.status_code == 202, response.text
        batch = response.json()
        await asyncio.wait_for(adapter.entered.wait(), 2)
        assert adapter.calls == 1
        assert (
            await client.get(f"/v1/batches/{batch['id']}", headers=bob_headers())
        ).status_code == 404
    completed = FakeAdapter()
    restarted = HarnessService(database, tmp_path, builder(completed), max_workers=1)
    async with client_for(restarted, monkeypatch) as client:
        runs = [await terminal(client, run["id"]) for run in batch["runs"]]
        assert [run["state"] for run in runs] == ["interrupted", "completed", "completed"]
        assert completed.calls == 2
        assert (await client.get(f"/v1/batches/{batch['id']}")).json()["complete"] is True


async def test_single_process_lease_fails_without_disrupting_owner(tmp_path):
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    duplicate = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    await service.start()
    try:
        with pytest.raises(RuntimeError, match="Another Harness server"):
            await duplicate.start()
        run = await service.submit("alice", RunSubmission(prompt="still running"))
        async with asyncio.timeout(3):
            async for _ in service.events("alice", run["id"]):
                pass
        assert (await service.store.run("alice", run["id"]))["state"] == "completed"
    finally:
        await service.close()


async def test_mcp_sdk_mount_retains_caller_identity_and_explicit_exposure(tmp_path, monkeypatch):
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    async with client_for(
        service, monkeypatch, expose_mcp=True, mcp_operations=("submit", "run_status", "sessions")
    ) as client:
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-11-25",
        }

        async def rpc(method, params, request_id=1, extra=None):
            return await client.post(
                "/mcp/",
                headers={**headers, **(extra or {})},
                json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            )

        initialized = await rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "offline-test", "version": "1"},
            },
        )
        assert initialized.status_code == 200, initialized.text
        listing = await rpc("tools/list", {})
        assert {tool["name"] for tool in listing.json()["result"]["tools"]} == {
            "submit",
            "run_status",
            "sessions",
        }
        result = await rpc(
            "tools/call", {"name": "submit", "arguments": {"prompt": "MCP identity scope"}}
        )
        assert result.status_code == 200, result.text
        assert not result.json()["result"].get("isError"), result.text
        run = result.json()["result"]["structuredContent"]
        await terminal(client, run["id"])
        stolen = await rpc(
            "tools/call",
            {"name": "run_status", "arguments": {"run_id": run["id"]}},
            extra=bob_headers(),
        )
        assert stolen.json()["result"]["isError"] is True
        assert run["session_id"] not in stolen.text
        missing_auth = await rpc("tools/list", {}, extra={"Authorization": ""})
        assert missing_auth.status_code == 401
        no_exposure = await rpc(
            "tools/call", {"name": "tool_run", "arguments": {"name": "shell", "arguments": {}}}
        )
        assert no_exposure.json()["result"]["isError"] is True


async def test_batch_creation_rolls_back_whole_batch_on_session_conflict(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter), max_workers=1)
    async with client_for(service, monkeypatch) as client:
        run = (await client.post("/v1/runs", json={"prompt": "first"})).json()
        await asyncio.wait_for(adapter.entered.wait(), 2)
        response = await client.post(
            "/v1/batches",
            json={
                "runs": [
                    {"prompt": "must roll back"},
                    {"prompt": "conflict", "session_id": run["session_id"]},
                ]
            },
        )
        assert response.status_code == 409
        assert await service.store.rows("SELECT id FROM api_batches") == []
        assert len(await service.store.rows("SELECT id FROM api_runs")) == 1
        assert len(await service.store.rows("SELECT id FROM api_sessions")) == 1


async def test_media_validation_and_persistence_use_core_schema(tmp_path, monkeypatch):
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    async with client_for(service, monkeypatch) as client:
        invalid = await client.post(
            "/v1/runs",
            json={
                "prompt": "bad media",
                "attachments": [
                    {"kind": "image", "mime_type": "image/png", "url": "file:///etc/passwd"}
                ],
            },
        )
        assert invalid.status_code == 422
        assert "file:///etc/passwd" not in invalid.text
        response = await client.post(
            "/v1/runs",
            json={
                "prompt": "view",
                "attachments": [{"kind": "image", "mime_type": "image/png", "data": "aGVsbG8="}],
            },
        )
        assert response.status_code == 202, response.text
        run = await terminal(client, response.json()["id"])
        # Fake provider may reject image capability; either way media is never silently dropped.
        messages = await service.messages("alice", run["session_id"])
        assert messages[0]["attachments"][0]["data"] == "aGVsbG8="


async def test_queued_cancel_does_not_start_model(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter), max_workers=1)
    async with client_for(service, monkeypatch) as client:
        first = (await client.post("/v1/runs", json={"prompt": "first"})).json()
        await asyncio.wait_for(adapter.entered.wait(), 2)
        queued = (await client.post("/v1/runs", json={"prompt": "queued"})).json()
        assert (await client.post(f"/v1/runs/{queued['id']}/cancel")).json()["state"] == "cancelled"
        await client.post(f"/v1/runs/{first['id']}/cancel")
        await asyncio.sleep(0.01)
        assert adapter.calls == 1


async def test_static_ui_is_public_but_all_data_requires_bearer(tmp_path, monkeypatch):
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    async with client_for(service, monkeypatch) as client:
        for path, expected in (
            ("/", "Connect to workspace"),
            ("/ui/app.js", "harnessConnect"),
            ("/ui/app.css", "color-scheme"),
        ):
            response = await client.get(path, headers={"Authorization": ""})
            assert response.status_code == 200 and expected in response.text
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
            assert "alice-token" not in response.text
        for path in ("/v1/runs", "/v1/batches", "/v1/sessions", "/v1/approvals", "/v1/health"):
            assert (await client.get(path, headers={"Authorization": ""})).status_code == 401
        assert (
            await client.get("/ui/../../api.db", headers={"Authorization": ""})
        ).status_code == 401
        await client.post("/v1/batches", json={"runs": [{"prompt": "first"}]})
        assert (await client.get("/v1/runs", headers=bob_headers())).json()["runs"] == []
        assert (await client.get("/v1/batches", headers=bob_headers())).json()["batches"] == []


async def test_denied_tool_action_cannot_be_resumed_as_success(tmp_path, monkeypatch):
    tool = RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(FakeAdapter(), tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch) as client:
        run = (
            await client.post(
                "/v1/tool-runs", json={"name": "record", "arguments": {"text": "denied"}}
            )
        ).json()
        assert (await terminal(client, run["id"]))["state"] == "paused"
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": False})
        response = await client.post(f"/v1/runs/{run['id']}/resume", json={})
        assert response.status_code == 409 and "denied" in response.text
        assert tool.values == []
