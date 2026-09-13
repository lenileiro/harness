from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest
from a2a.utils.errors import InvalidParamsError

from harness.server import HarnessService
from harness.server.a2a_callbacks import CallbackGrant, CallbackHTTP, CallbackManager

from .test_a2a import rpc, send
from .test_service import FakeAdapter, bob_headers, builder, client_for, terminal

URL = "https://receiver.example/hooks/a2a"
SECRET = "callback-signing-secret-" + "s" * 40
BEARER = "receiver-only-token"


async def public_dns(host, port):
    assert host == "receiver.example" and port == 443
    return ["93.184.216.34"]


def manager_for(service, handler, *, attempts=5, grants=True):
    manager = CallbackManager(
        service,
        [
            CallbackGrant(
                owner="alice", url=URL, secret_env="CALLBACK_SECRET", bearer_env="CALLBACK_BEARER"
            )
        ],
        environment={"CALLBACK_SECRET": SECRET, "CALLBACK_BEARER": BEARER},
        http=CallbackHTTP(transport=httpx.MockTransport(handler), resolver=public_dns),
        max_attempts=attempts,
    )
    if not grants:
        manager.grants.clear()
    service.a2a_callbacks = manager
    return manager


def configuration(task_id="", identifier="notify"):
    return {"id": identifier, "taskId": task_id, "url": URL, "token": "correlation-123"}


async def submit_completed(client, *, inline=False):
    params = send()
    if inline:
        params["configuration"]["taskPushNotificationConfig"] = configuration()
    reply = await rpc(client, "SendMessage", params)
    assert "result" in reply, reply
    task = reply["result"]["task"]
    await terminal(client, task["metadata"]["harness_run_id"])
    return task, params


async def drain(manager):
    for index in range(10):
        await manager.tick(moment=time.time() + index * 4000)


async def test_owned_official_crud_inline_atomic_retry_and_exact_signed_payload(
    tmp_path, monkeypatch
):
    requests = []

    def receive(request):
        requests.append(request)
        return httpx.Response(204)

    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    manager = manager_for(service, receive)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        assert (await client.get("/.well-known/agent-card.json")).json()["capabilities"][
            "pushNotifications"
        ]
        task, original = await submit_completed(client, inline=True)
        task_id = task["id"]
        retry = await rpc(client, "SendMessage", original)
        assert retry["result"]["task"]["id"] == task_id and adapter.calls == 1
        expected = {"taskId": task_id, "id": "notify"}
        saved = (await rpc(client, "GetTaskPushNotificationConfig", expected))["result"]
        assert saved["authentication"] == {"scheme": "Bearer"}
        assert "credentials" not in json.dumps(saved)
        created = (await rpc(client, "CreateTaskPushNotificationConfig", configuration(task_id)))[
            "result"
        ]
        assert created == saved
        listed = (await rpc(client, "ListTaskPushNotificationConfigs", {"taskId": task_id}))[
            "result"
        ]
        assert listed["configs"] == [saved]
        for method, params in (
            ("GetTaskPushNotificationConfig", expected),
            ("DeleteTaskPushNotificationConfig", expected),
            ("ListTaskPushNotificationConfigs", {"taskId": task_id}),
            ("CreateTaskPushNotificationConfig", configuration(task_id)),
        ):
            assert (await rpc(client, method, params, headers=bob_headers()))["error"][
                "code"
            ] == -32001
        await drain(manager)
        assert len(requests) == 3
        states = []
        for request in requests:
            assert request.url.host == "93.184.216.34"
            assert request.headers["Host"] == "receiver.example"
            assert request.extensions["sni_hostname"] == "receiver.example"
            assert request.headers["Authorization"] == "Bearer " + BEARER
            assert request.headers["X-A2A-Notification-Token"] == "correlation-123"
            # Exact pinned Hermes convention: raw-body SHA256 HMAC, lowercase hex, no prefix.
            assert (
                request.headers["X-A2A-Signature"]
                == hmac.new(SECRET.encode(), request.content, hashlib.sha256).hexdigest()
            )
            payload = json.loads(request.content)["task"]
            assert (
                payload["metadata"]["harness_delivery_id"] == request.headers["X-A2A-Delivery-ID"]
            )
            assert payload["id"] == task_id
            states.append(payload["status"]["state"])
        assert states == ["TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED"]
        assert (
            json.loads(requests[-1].content)["task"]["artifacts"][0]["parts"][0]["text"]
            == "Hello from the test model"
        )
        diagnostics = (await client.get("/v1/a2a/callback-deliveries")).json()
        assert all(row["state"] == "delivered" for row in diagnostics["deliveries"])
        assert SECRET not in json.dumps(diagnostics) and BEARER not in json.dumps(diagnostics)
        assert (await client.get("/v1/a2a/callback-deliveries", headers=bob_headers())).json()[
            "deliveries"
        ] == []
        await rpc(client, "DeleteTaskPushNotificationConfig", expected)
        assert "error" in await rpc(client, "GetTaskPushNotificationConfig", expected)


@pytest.mark.parametrize(
    "bad",
    [
        {"url": "https://unapproved.example/hooks"},
        {"url": "https://127.0.0.1/hooks"},
        {"authentication": {"scheme": "Bearer", "credentials": "inline-secret"}},
        {"token": " padded "},
        {"taskId": "somebody-elses-task"},
    ],
)
async def test_invalid_inline_subscription_cannot_enqueue_work(tmp_path, monkeypatch, bad):
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    manager = manager_for(service, lambda request: pytest.fail("Unexpected callback"))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        params = send()
        params["configuration"]["taskPushNotificationConfig"] = {**configuration(), **bad}
        assert (await rpc(client, "SendMessage", params))["error"]["code"] == -32602
        assert (await client.get("/v1/runs")).json()["runs"] == []
        assert not await service.store.rows("SELECT * FROM api_a2a_callbacks")
        assert adapter.calls == 0


async def test_inline_callback_failure_rolls_back_entire_executable_binding(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    manager = manager_for(service, lambda request: pytest.fail("Unexpected callback"))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        async with service.store.connection() as db:
            await db.execute(
                "CREATE TRIGGER reject_callback BEFORE INSERT ON api_a2a_callbacks BEGIN SELECT RAISE(ABORT, 'test callback persistence failure'); END"
            )
            await db.commit()
        params = send()
        params["configuration"]["taskPushNotificationConfig"] = configuration()
        assert "error" in await rpc(client, "SendMessage", params)
        for table in (
            "api_runs",
            "api_events",
            "api_a2a_tasks",
            "api_a2a_messages",
            "api_a2a_task_runs",
            "api_a2a_callbacks",
        ):
            assert not await service.store.rows("SELECT * FROM " + table)
        assert adapter.calls == 0


async def test_lost_ack_restart_retries_same_signed_body_without_replaying_agent(
    tmp_path, monkeypatch
):
    requests = []

    def interrupted(request):
        requests.append(request)
        raise httpx.ReadTimeout("Receiver may already have accepted this notification")

    path = tmp_path / "api.db"
    adapter = FakeAdapter()
    service = HarnessService(path, tmp_path, builder(adapter))
    manager = manager_for(service, interrupted)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        task, original = await submit_completed(client, inline=True)
        await manager.tick()
        first = (await manager.deliveries("alice"))[-1]
        assert first["state"] == "pending" and first["attempts"] == 1
        # No later task status overtakes this outstanding notification.
        await manager.tick(moment=first["next_attempt"] - 1)
        assert len(requests) == 1
        # Simulate crash after a persisted claim, before acknowledgement persisted.
        async with service.store.connection() as db:
            await db.execute(
                "UPDATE api_a2a_callback_deliveries SET state='sending',next_attempt=? WHERE id=?",
                (time.time() + 10000, first["id"]),
            )
            await db.commit()
    restored_adapter = FakeAdapter()
    restored = HarnessService(path, tmp_path, builder(restored_adapter))
    restored_manager = manager_for(
        restored, lambda request: requests.append(request) or httpx.Response(204)
    )
    async with client_for(restored, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await restored_manager.close()
        assert (await restored_manager.deliveries("alice"))[-1]["state"] == "pending"
        await restored_manager.tick(moment=time.time() + 11000)
        assert requests[0].content == requests[1].content
        assert requests[0].headers["X-A2A-Signature"] == requests[1].headers["X-A2A-Signature"]
        assert requests[0].headers["X-A2A-Delivery-ID"] == requests[1].headers["X-A2A-Delivery-ID"]
        assert (await rpc(client, "SendMessage", original))["result"]["task"]["id"] == task["id"]
        assert restored_adapter.calls == 0


async def test_retry_limits_deletion_and_removed_operator_grant(tmp_path, monkeypatch):
    requests = []
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    manager = manager_for(
        service,
        lambda request: (
            requests.append(request) or httpx.Response(503, headers={"Retry-After": "60"})
        ),
        attempts=2,
    )
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        task, _ = await submit_completed(client)
        # A late subscription replays the recorded owned task status history.
        await rpc(client, "CreateTaskPushNotificationConfig", configuration(task["id"]))
        now = time.time() + 1
        await manager.tick(moment=now)
        first = (await manager.deliveries("alice"))[-1]
        assert first["next_attempt"] >= now + 60
        await manager.tick(moment=now + 100)
        first = (await manager.deliveries("alice"))[-1]
        assert first["attempts"] == 2 and first["state"] == "failed"
        manager.grants.clear()
        await manager.tick(moment=now + 200)
        assert len(requests) == 2
        assert any(
            row["error"] == "Callback endpoint grant was removed"
            for row in await manager.deliveries("alice")
        )
        await rpc(
            client, "DeleteTaskPushNotificationConfig", {"taskId": task["id"], "id": "notify"}
        )
        await manager.tick(moment=now + 300)
        assert len(requests) == 2
        assert any(row["state"] == "cancelled" for row in await manager.deliveries("alice"))


@pytest.mark.parametrize(
    "addresses",
    [["127.0.0.1"], ["93.184.216.34", "10.0.0.1"], ["::ffff:127.0.0.1"], ["224.0.0.1"], []],
)
async def test_private_mixed_and_multicast_dns_fail_before_any_http(addresses):
    async def resolve(host, port):
        return addresses

    http = CallbackHTTP(
        transport=httpx.MockTransport(lambda request: pytest.fail("No forbidden network request")),
        resolver=resolve,
    )
    with pytest.raises(InvalidParamsError):
        await http.post(URL, b"{}", {})


async def test_redirect_is_not_followed_and_cancellation_leaves_durable_claim(
    tmp_path, monkeypatch
):
    requests = []
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    manager = manager_for(
        service,
        lambda request: (
            requests.append(request)
            or httpx.Response(307, headers={"Location": "http://127.0.0.1/private"})
        ),
    )
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        await submit_completed(client, inline=True)
        await manager.tick()
        assert len(requests) == 1 and (await manager.deliveries("alice"))[-1]["state"] == "failed"
        entered = asyncio.Event()

        async def block(request) -> httpx.Response:
            entered.set()
            return await asyncio.Future[httpx.Response]()

        manager.http = CallbackHTTP(transport=httpx.MockTransport(block), resolver=public_dns)
        dispatch = asyncio.create_task(manager.tick())
        await asyncio.wait_for(entered.wait(), 2)
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        assert any(
            row["state"] == "sending" and row["attempts"] == 1
            for row in await manager.deliveries("alice")
        )
        await manager.delete(
            "alice", (await service.store.rows("SELECT id FROM api_a2a_tasks"))[0]["id"], "notify"
        )
        await manager.start()
        await manager.close()
        assert all(
            row["state"] in {"cancelled", "failed"} for row in await manager.deliveries("alice")
        )


def test_secret_reference_errors_do_not_disclose_secret_values(tmp_path):
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    grant = CallbackGrant(owner="alice", url=URL, secret_env="SIGNING")
    with pytest.raises(ValueError, match="environment reference") as error:
        CallbackManager(service, [grant], environment={"SIGNING": "private"})
    assert "private" not in str(error.value)
    with pytest.raises(InvalidParamsError):
        CallbackHTTP()  # construction does not resolve DNS
        from harness.server.a2a_callbacks import callback_url

        callback_url("https://user:password@public.example/")


async def test_legacy_wire_registration_crud_signature_and_restart(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "api.db"
    service = HarnessService(path, tmp_path, builder(FakeAdapter()))
    manager = manager_for(service, lambda request: requests.append(request) or httpx.Response(503))
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        legacy = {
            "message": {
                "messageId": "old",
                "role": "user",
                "parts": [{"kind": "text", "text": "hello"}],
            },
            "configuration": {
                "blocking": False,
                "pushNotificationConfig": {"id": "legacy", "url": URL},
            },
        }
        response = await rpc(client, "message/send", legacy)
        assert "result" in response, response
        task = response["result"]
        await terminal(client, task["metadata"]["harness_run_id"])
        query = {"id": task["id"], "pushNotificationConfigId": "legacy"}
        saved = await rpc(client, "tasks/pushNotificationConfig/get", query)
        assert saved["result"]["pushNotificationConfig"]["id"] == "legacy"
        created = await rpc(
            client,
            "tasks/pushNotificationConfig/set",
            {"taskId": task["id"], "pushNotificationConfig": {"id": "legacy", "url": URL}},
        )
        assert created["result"] == saved["result"]
        listed = await rpc(client, "tasks/pushNotificationConfig/list", {"id": task["id"]})
        assert len(listed["result"]) == 1
        assert "error" in await rpc(
            client, "GetTaskPushNotificationConfig", {"taskId": task["id"], "id": "legacy"}
        )
        assert "error" in await rpc(
            client, "CreateTaskPushNotificationConfig", configuration(task["id"], "legacy")
        )
        foreign = await rpc(
            client, "tasks/pushNotificationConfig/get", query, headers=bob_headers()
        )
        assert "error" in foreign
        await manager.tick()
        assert len(requests) == 1
        first = (await manager.deliveries("alice"))[-1]
        async with service.store.connection() as db:
            await db.execute(
                "UPDATE api_a2a_callback_deliveries SET next_attempt=? WHERE id=?",
                (time.time() + 10000, first["id"]),
            )
            await db.commit()
    adapter = FakeAdapter()
    restored = HarnessService(path, tmp_path, builder(adapter))
    manager = manager_for(restored, lambda request: requests.append(request) or httpx.Response(204))
    async with client_for(restored, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await manager.close()
        for index in range(4):
            await manager.tick(moment=time.time() + 11000 + index)
        assert len(requests) == 4 and requests[0].content == requests[1].content
        for request in requests:
            assert request.headers["A2A-Version"] == "0.3"
            assert (
                request.headers["X-A2A-Signature"]
                == hmac.new(SECRET.encode(), request.content, hashlib.sha256).hexdigest()
            )
            body = json.loads(request.content)
            assert body["kind"] == "task" and body["id"] == task["id"] and "task" not in body
            assert body["metadata"]["harness_delivery_id"] == request.headers["X-A2A-Delivery-ID"]
        final = json.loads(requests[-1].content)
        assert final["status"]["state"] == "completed"
        assert final["artifacts"][0]["parts"][0]["kind"] == "text"
        assert (await rpc(client, "message/send", legacy))["result"]["id"] == task["id"]
        assert adapter.calls == 0
        removed = await rpc(client, "tasks/pushNotificationConfig/delete", query)
        assert "error" not in removed
        assert "error" in await rpc(client, "tasks/pushNotificationConfig/get", query)


@pytest.mark.parametrize("version", ["0.3", "1.0"])
def test_callback_canonical_unicode_bytes_match_pinned_hermes_signer(version):
    from a2a.types import a2a_pb2 as proto

    from harness.server.a2a_callbacks import notification_payload

    task = proto.Task(
        id="task-unicode",
        context_id="context",
        status=proto.TaskStatus(state=proto.TASK_STATE_COMPLETED),
        metadata={"note": "Tere, maailm! Häid pühi 日本語"},
    )
    body = notification_payload(task, version).encode("utf-8")
    # The pinned sign_push_payload function computes precisely these bytes.
    canonical = json.dumps(json.loads(body), sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert body == canonical and "日本語".encode() in body
    assert (
        hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        == hmac.new(SECRET.encode(), canonical, hashlib.sha256).hexdigest()
    )


async def test_existing_callback_schema_migrates_without_rewriting_config(tmp_path):
    import sqlite3

    from harness.server.store import ServiceStore

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE api_a2a_callbacks (owner TEXT, task_id TEXT, id TEXT, config TEXT, last_seq INTEGER, active INTEGER, PRIMARY KEY(owner,task_id,id))"
        )
        db.execute(
            "INSERT INTO api_a2a_callbacks VALUES ('alice','task','notify',?,12,1)",
            (json.dumps(configuration("task")),),
        )
    store = ServiceStore(path)
    try:
        await store.start()
        records = await store.rows("SELECT * FROM api_a2a_callbacks")
        assert records[0]["protocol"] == "1.0" and records[0]["last_seq"] == 12
        assert records[0]["config"] == json.dumps(configuration("task"))
    finally:
        await store.close()
