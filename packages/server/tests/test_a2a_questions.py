from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from harness.core.approval import PendingApproval

from .test_a2a import rpc, send
from .test_questions import Adapter, client_for, finished, service_for


async def start_question(client):
    task = (await rpc(client, "SendMessage", send()))["result"]["task"]
    run = await finished(client, task["metadata"]["harness_run_id"])
    assert run["state"] == "paused", run
    detail = (await rpc(client, "GetTask", {"id": task["id"]}))["result"]
    assert "Which outputs?" in detail["status"]["message"]["parts"][0]["text"]
    assert "Human approval is required" not in detail["status"]["message"]["parts"][0]["text"]
    [question] = (await client.get("/v1/questions")).json()["questions"]
    assert detail["metadata"]["harness_question_id"] == question["id"]
    return task, question


def answer(task, message_id, text):
    return send(message_id, text=text, taskId=task["id"], contextId=task["contextId"])


async def test_partial_answer_duplicate_concurrency_restart_and_final_resume(tmp_path, monkeypatch):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task, _question = await start_question(client)
        partial = answer(task, "partial", "1, 3, Custom")
        responses = await asyncio.gather(*(rpc(client, "SendMessage", partial) for _ in range(4)))
        assert all("result" in reply for reply in responses), responses
        assert all(
            reply["result"]["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
            for reply in responses
        )
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["answers"] == {"q0": ["Report", "Table", "Custom"]}
        assert len(adapter.calls) == 1
        assert "error" in await rpc(client, "SendMessage", answer(task, "partial", "different"))
        assert "error" in await rpc(
            client,
            "SendMessage",
            answer(task, "foreign", "intruder"),
            headers={"Authorization": "Bearer other-" + 40 * "t"},
        )
        assert "error" in await rpc(client, "SendMessage", answer(task, "invalid", '{"q1": 99}'))
        assert "error" in await rpc(
            client, "SendMessage", answer(task, "change", '{"q0": ["Chart"]}')
        )
    restored_adapter = Adapter()
    restored = service_for(tmp_path, restored_adapter)
    async with client_for(restored, monkeypatch, a2a_url="http://localhost/a2a") as client:
        await rpc(client, "SendMessage", partial)
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["answers"] == {"q0": ["Report", "Table", "Custom"]}
        final = answer(task, "final", "Engineering team")
        responses = await asyncio.gather(*(rpc(client, "SendMessage", final) for _ in range(4)))
        assert all("result" in response for response in responses), responses
        response = responses[0]
        resumed = response["result"]["task"]
        assert resumed["id"] == task["id"]
        assert (await finished(client, resumed["metadata"]["harness_run_id"]))[
            "state"
        ] == "completed"
        repeated = await rpc(client, "SendMessage", final)
        assert repeated["result"]["task"]["id"] == task["id"]
        assert len(restored_adapter.calls) == 1
        results = [
            message
            for message in restored_adapter.calls[-1]["messages"]
            if message.role == "tool" and message.name == "clarify"
        ]
        assert json.loads(results[0].content)["responses"][1]["user_response"] == "Engineering team"
        assert (await client.get("/v1/questions")).json()["questions"] == []


async def test_answer_and_receipt_rollback_is_atomic(tmp_path, monkeypatch):
    service = service_for(tmp_path, Adapter())
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task, _question = await start_question(client)
        async with service.store.connection() as db:
            await db.execute(
                "CREATE TRIGGER reject_answer_receipt BEFORE INSERT ON api_a2a_answer_receipts BEGIN SELECT RAISE(ABORT, 'test receipt persistence failure'); END"
            )
            await db.commit()
        reply = await rpc(client, "SendMessage", answer(task, "attempt", "Report"))
        assert "error" in reply
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["answers"] == {}
        assert not await service.store.rows("SELECT * FROM api_a2a_answer_receipts")
        async with service.store.connection() as db:
            await db.execute("DROP TRIGGER reject_answer_receipt")
            await db.commit()
        assert "result" in await rpc(client, "SendMessage", answer(task, "attempt", "Report"))
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["answers"] == {"q0": ["Report"]}


async def test_complete_answer_survives_enqueue_failure_and_retry_never_grants_approval(
    tmp_path, monkeypatch
):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task, _question = await start_question(client)
        await service.storage.create_approval(
            PendingApproval(
                id="separate-approval",
                session_id=task["contextId"],
                tool_call_id="effect",
                tool_name="record",
                arguments={},
            )
        )
        final = answer(task, "complete", '{"q0":["Report","Custom"],"q1":"Team"}')
        blocked = await rpc(client, "SendMessage", final)
        assert "error" in blocked and "approvals" in blocked["error"]["message"]
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["status"] == "answered"
        approval = await service.storage.get_approval("separate-approval")
        assert approval is not None and approval.status == "pending"
        assert len(adapter.calls) == 1
    restored = service_for(tmp_path, adapter)
    async with client_for(restored, monkeypatch, a2a_url="http://localhost/a2a") as client:
        # Same accepted message still cannot bypass approval after restart.
        assert "error" in await rpc(client, "SendMessage", final)
        response = await client.post(
            "/v1/approvals/separate-approval/resolve", json={"granted": False}
        )
        assert response.status_code == 200
        retried = await rpc(client, "SendMessage", final)
        assert "result" in retried, retried
        await finished(client, retried["result"]["task"]["metadata"]["harness_run_id"])
        await rpc(client, "SendMessage", final)
        assert len(adapter.calls) == 2


@pytest.mark.parametrize("end", ["expired", "cancelled"])
async def test_expired_or_cancelled_question_continues_without_reinterpreting_text(
    tmp_path, monkeypatch, end
):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task, question = await start_question(client)
        await rpc(client, "SendMessage", answer(task, "partial", "Report"))
        if end == "cancelled":
            response = await client.post(f"/v1/questions/{question['id']}/cancel")
            assert response.status_code == 200
        else:
            async with service.store.connection() as db:
                expired = datetime.now(UTC) - timedelta(seconds=1)
                await db.execute(
                    "UPDATE clarifications SET expires_at=?,payload=json_set(payload,'$.expires_at',?) WHERE id=?",
                    (expired.timestamp(), expired.isoformat(), question["id"]),
                )
                await db.commit()
        result = await rpc(client, "SendMessage", answer(task, "continue", "Please continue"))
        assert "result" in result, result
        await finished(client, result["result"]["task"]["metadata"]["harness_run_id"])
        output = next(
            message
            for message in adapter.calls[-1]["messages"]
            if message.role == "tool" and message.name == "clarify"
        )
        value = json.loads(output.content)
        assert value["responses"][0]["user_response"] == ["Report"]
        assert value["responses"][1]["user_response"] == ""
        assert value["timed_out" if end == "expired" else "cancelled"] is True


async def test_legacy_text_json_answer_uses_same_owned_pipeline(tmp_path, monkeypatch):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch, a2a_url="http://localhost/a2a") as client:
        task, _question = await start_question(client)
        legacy = {
            "message": {
                "messageId": "legacy-answer",
                "role": "user",
                "taskId": task["id"],
                "contextId": task["contextId"],
                "parts": [{"kind": "text", "text": '{"q0":["Chart"],"q1":"Stakeholders"}'}],
            },
            "configuration": {"blocking": False},
        }
        reply = await rpc(client, "message/send", legacy)
        assert "result" in reply, reply
        await finished(client, reply["result"]["metadata"]["harness_run_id"])
        assert (await rpc(client, "message/send", legacy))["result"]["status"][
            "state"
        ] == "completed"
        assert len(adapter.calls) == 2
