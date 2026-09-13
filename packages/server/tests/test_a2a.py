from __future__ import annotations

import asyncio
import json

import pytest
from a2a.server.context import ServerCallContext
from a2a.types import a2a_pb2 as proto

from harness.core.events import Done
from harness.core.schemas import Message, ToolCall
from harness.server import HarnessService
from harness.server.a2a import Caller, HarnessA2AHandler

from .test_service import FakeAdapter, RecordingTool, bob_headers, builder, client_for, terminal


async def rpc(client, method, params, **kwargs):
    kwargs["headers"] = {
        "A2A-Version": "0.3" if "/" in method else "1.0",
        **kwargs.get("headers", {}),
    }
    response = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "request", "method": method, "params": params},
        **kwargs,
    )
    assert response.status_code == 200, response.text
    return response.json()


def send(message_id="first", *, text="hello", **kwargs):
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"text": text}],
            **kwargs,
        },
        "configuration": {"returnImmediately": True},
    }


async def test_official_jsonrpc_auth_card_retry_scope_and_restart(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    path = tmp_path / "api.db"
    service = HarnessService(path, tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        assert (
            await client.get("/.well-known/agent-card.json", headers={"Authorization": ""})
        ).status_code == 401
        card = (await client.get("/.well-known/agent-card.json")).json()
        assert card["supportedInterfaces"][0]["url"] == "http://localhost/a2a"
        assert "bearer" in card["securitySchemes"]
        assert (
            await client.post("/a2a", json={}, headers={"Origin": "http://evil.invalid"})
        ).status_code == 403
        responses = await asyncio.gather(*(rpc(client, "SendMessage", send()) for _ in range(4)))
        tasks = [response["result"]["task"] for response in responses]
        assert len({task["id"] for task in tasks}) == 1
        task = tasks[0]
        run_id = task["metadata"]["harness_run_id"]
        await terminal(client, run_id)
        assert adapter.calls == 1
        mismatch = await rpc(client, "SendMessage", send(text="different"))
        assert mismatch["error"]["code"] == -32602
        for method in ("GetTask", "CancelTask", "SubscribeToTask"):
            assert (await rpc(client, method, {"id": task["id"]}, headers=bob_headers()))["error"][
                "code"
            ] == -32001
        assert (await rpc(client, "ListTasks", {}, headers=bob_headers()))["result"].get(
            "tasks", []
        ) == []
        result = (await rpc(client, "GetTask", {"id": task["id"], "historyLength": 10}))["result"]
        assert result["status"]["state"] == "TASK_STATE_COMPLETED"
        assert result["artifacts"][0]["parts"][0]["text"] == "Hello from the test model"
        assert [message["role"] for message in result["history"]] == ["ROLE_USER", "ROLE_AGENT"]
        continued = (await rpc(client, "SendMessage", send("next", contextId=task["contextId"])))[
            "result"
        ]["task"]
        assert continued["id"] != task["id"] and continued["contextId"] == task["contextId"]
        await terminal(client, continued["metadata"]["harness_run_id"])
    restarted_adapter = FakeAdapter()
    restarted = HarnessService(path, tmp_path, builder(restarted_adapter))
    async with client_for(restarted, monkeypatch, a2a_url="http://localhost/a2a") as client:
        retried = (await rpc(client, "SendMessage", send()))["result"]["task"]
        assert retried["id"] == task["id"] and restarted_adapter.calls == 0
        page = (await rpc(client, "ListTasks", {"pageSize": 1}))["result"]
        assert page["totalSize"] == 2 and page["nextPageToken"]
        next_page = (
            await rpc(client, "ListTasks", {"pageSize": 1, "pageToken": page["nextPageToken"]})
        )["result"]
        assert next_page["tasks"][0]["id"] != page["tasks"][0]["id"]


async def test_sdk_stream_compatibility_and_actual_cancel(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        old = {
            "message": {
                "messageId": "old",
                "role": "user",
                "parts": [{"kind": "text", "text": "wait"}],
            },
            "configuration": {"blocking": False},
        }
        result = (await rpc(client, "message/send", old))["result"]
        assert result["kind"] == "task"
        await asyncio.wait_for(adapter.entered.wait(), 2)
        cancelled = (await rpc(client, "CancelTask", {"id": result["id"]}))["result"]
        assert cancelled["status"]["state"] == "TASK_STATE_CANCELED" and adapter.cancelled.is_set()
        # An actual SDK SSE response is replayable after completion.
        response = await client.post(
            "/a2a",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "SubscribeToTask",
                "params": {"id": result["id"]},
            },
        )
        assert response.headers["content-type"].startswith("text/event-stream")
        data = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert data[-1]["result"]["task"]["status"]["state"] == "TASK_STATE_CANCELED"
        assert (
            await client.post(f"/v1/runs/{cancelled['metadata']['harness_run_id']}/resume", json={})
        ).status_code == 409


async def test_disconnect_stops_observation_without_cancelling_work(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        handler = HarnessA2AHandler(service)
        stream = handler.on_subscribe_to_task(
            proto.SubscribeToTaskRequest(id=task["id"]), ServerCallContext(user=Caller("alice"))
        )
        await anext(stream)
        await stream.aclose()
        await asyncio.wait_for(adapter.entered.wait(), 2)
        assert not adapter.cancelled.is_set() and adapter.calls == 1
        assert adapter.block is not None
        adapter.block.set()
        assert (await terminal(client, task["metadata"]["harness_run_id"]))["state"] == "completed"


class ActionAdapter(FakeAdapter):
    async def stream(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="action", name="record", arguments={"text": "effect"})],
                )
            )
        else:
            yield Done(final_message=Message(role="assistant", content="Action complete"))


async def test_approval_input_required_and_continuation_same_task(tmp_path, monkeypatch):
    adapter, tool = ActionAdapter(), RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(adapter, tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        await terminal(client, task["metadata"]["harness_run_id"])
        paused = (await rpc(client, "GetTask", {"id": task["id"]}))["result"]
        assert paused["status"]["state"] == "TASK_STATE_INPUT_REQUIRED" and tool.values == []
        continuation = send("continue", taskId=task["id"], contextId=task["contextId"])
        assert (await rpc(client, "SendMessage", continuation))["error"]["code"] == -32602
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        resumed = (await rpc(client, "SendMessage", continuation))["result"]["task"]
        assert resumed["id"] == task["id"]
        await terminal(client, resumed["metadata"]["harness_run_id"])
        assert tool.values == ["effect"]
        await rpc(client, "SendMessage", continuation)
        assert tool.values == ["effect"]


@pytest.mark.parametrize("grant", [False, True])
async def test_paused_cancel_revokes_pending_and_unclaimed_grants(tmp_path, monkeypatch, grant):
    adapter, tool = ActionAdapter(), RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(adapter, tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        run_id = task["metadata"]["harness_run_id"]
        await terminal(client, run_id)
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        if grant:
            await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        cancelled = (await rpc(client, "CancelTask", {"id": task["id"]}))["result"]
        assert cancelled["status"]["state"] == "TASK_STATE_CANCELED" and tool.values == []
        record = await service.storage.get_approval(approval["id"])
        assert record is not None and record.status == "denied"
        assert not await service.storage.claim_replay(approval["id"], session_id=task["contextId"])
        assert (await client.post(f"/v1/runs/{run_id}/resume", json={})).status_code == 409


async def test_claimed_effect_cannot_be_reported_cancelled(tmp_path, monkeypatch):
    adapter, tool = ActionAdapter(), RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(adapter, tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        await terminal(client, task["metadata"]["harness_run_id"])
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        assert await service.storage.claim_replay(approval["id"], session_id=task["contextId"])
        assert (await rpc(client, "CancelTask", {"id": task["id"]}))["error"]["code"] == -32002
        assert (await rpc(client, "GetTask", {"id": task["id"]}))["result"]["status"][
            "state"
        ] == "TASK_STATE_INPUT_REQUIRED"


async def test_invalid_content_and_unsupported_callbacks_never_enqueue(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        invalid = [send("data"), send("file"), send("tenant"), send("push")]
        invalid[0]["message"]["parts"] = [{"data": {"unsafe": "control"}}]
        invalid[1]["message"]["parts"] = [{"url": "file:///secret", "mediaType": "text/plain"}]
        invalid[2]["tenant"] = "bob"
        invalid[3]["configuration"]["taskPushNotificationConfig"] = {"id": "callback"}
        for params in invalid:
            assert "error" in await rpc(client, "SendMessage", params)
        assert (await client.get("/v1/runs")).json()["runs"] == []
        assert adapter.calls == 0
        assert (
            await client.post("/a2a", content=b"{}", headers={"Content-Type": "text/plain"})
        ).status_code == 415
        assert (
            await client.post(
                "/a2a",
                content=b" " * (24 * 1024 * 1024 + 1),
                headers={"Content-Type": "application/json"},
            )
        ).status_code == 413


async def test_official_sdk_client_stream_and_media_round_trip(tmp_path, monkeypatch):
    from a2a.client.client import ClientCallContext
    from a2a.client.transports.jsonrpc import JsonRpcTransport
    from google.protobuf.json_format import ParseDict

    from harness.core.schemas import Capabilities

    class MediaAdapter(FakeAdapter):
        async def capabilities(self):
            return Capabilities(tool_use=True, input_media=["image"])

    adapter = MediaAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        card = ParseDict(
            (await client.get("/.well-known/agent-card.json")).json(), proto.AgentCard()
        )
        transport = JsonRpcTransport(client, card, "http://localhost/a2a")
        request = proto.SendMessageRequest(
            message=proto.Message(
                message_id="sdk",
                role=proto.ROLE_USER,
                parts=[
                    proto.Part(text="Describe the attached synthetic image"),
                    proto.Part(
                        raw=b"synthetic test image", media_type="image/png", filename="sample.png"
                    ),
                ],
            )
        )
        context = ClientCallContext(service_parameters={"A2A-Version": "1.0"})
        stream = [
            event async for event in transport.send_message_streaming(request, context=context)
        ]
        task = stream[-1].task
        assert task.status.state == proto.TASK_STATE_COMPLETED and task.artifacts
        saved = await transport.get_task(
            proto.GetTaskRequest(id=task.id, history_length=5), context=context
        )
        assert saved.history[0].parts[1].raw == b"synthetic test image"
        assert saved.history[0].parts[1].media_type == "image/png"
        assert adapter.calls == 1


async def test_a2a_inflight_restart_remains_failed_without_automatic_replay(tmp_path, monkeypatch):
    adapter = FakeAdapter(block=asyncio.Event())
    path = tmp_path / "api.db"
    service = HarnessService(path, tmp_path, builder(adapter))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        await asyncio.wait_for(adapter.entered.wait(), 2)
    after = FakeAdapter()
    restarted = HarnessService(path, tmp_path, builder(after))
    async with client_for(restarted, monkeypatch, a2a_url="http://localhost/a2a") as client:
        result = (await rpc(client, "SendMessage", send()))["result"]["task"]
        assert result["id"] == task["id"] and result["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            "stopped during" in result["status"]["message"]["parts"][0]["text"] and after.calls == 0
        )


async def test_a2a_identity_rollback_and_api_resume_updates_task(tmp_path, monkeypatch):
    import sqlite3

    from harness.server.a2a_store import A2ABinding

    adapter, tool = ActionAdapter(), RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(adapter, tool), exposed_tools=["record"]
    )
    original = A2ABinding.save

    async def fail(*args):
        raise sqlite3.IntegrityError("synthetic transaction failure")

    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        monkeypatch.setattr(A2ABinding, "save", fail)
        assert "error" in await rpc(client, "SendMessage", send())
        assert await service.store.rows("SELECT * FROM api_runs") == []
        assert await service.store.rows("SELECT * FROM api_a2a_messages") == []
        assert await service.store.rows("SELECT * FROM api_sessions") == []
        monkeypatch.setattr(A2ABinding, "save", original)
        task = (await rpc(client, "SendMessage", send()))["result"]["task"]
        run_id = task["metadata"]["harness_run_id"]
        await terminal(client, run_id)
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        resumed = (await client.post(f"/v1/runs/{run_id}/resume", json={})).json()
        await terminal(client, resumed["id"])
        result = (await rpc(client, "GetTask", {"id": task["id"]}))["result"]
        assert (
            result["metadata"]["harness_run_id"] == resumed["id"]
            and result["status"]["state"] == "TASK_STATE_COMPLETED"
        )
        assert tool.values == ["effect"]
