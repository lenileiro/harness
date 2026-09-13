import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from harness.core import Agent, Capabilities, Done, FailoverPolicy, Message, ToolCall, ToolRegistry
from harness.server import HarnessService, ServerAuth, create_app


class Adapter:
    name = "fixture"

    def __init__(self):
        self.calls = []

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        prior = [
            message
            for message in kwargs["messages"]
            if message.role == "tool" and message.name == "clarify"
        ]
        if not prior:
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="ask",
                            name="clarify",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Which outputs?",
                                        "choices": ["Report", "Chart", "Table"],
                                        "multi_select": True,
                                    },
                                    {"question": "Who is the audience?"},
                                ]
                            },
                        )
                    ],
                )
            )
        else:
            yield Done(
                final_message=Message(role="assistant", content="Answers received. Work complete.")
            )


def service_for(tmp_path, adapter, exposed=("clarify",)):
    def build(context):
        return Agent(
            adapters={"fixture": adapter},
            tools=ToolRegistry(),
            storage=context.storage,
            failover=FailoverPolicy(chain=["fixture"]),
            default_model="test",
            default_cwd=str(tmp_path),
        )

    return HarnessService(tmp_path / "service.db", tmp_path, build, exposed_tools=exposed)


@asynccontextmanager
async def client_for(service, monkeypatch, **options):
    monkeypatch.setenv("QUESTION_OWNER", "owner-" + 40 * "o")
    monkeypatch.setenv("QUESTION_OTHER", "other-" + 40 * "t")
    app = create_app(
        service,
        auth=ServerAuth(token_envs={"owner": "QUESTION_OWNER", "other": "QUESTION_OTHER"}),
        **options,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer owner-" + 40 * "o"},
        ) as client,
    ):
        yield client


async def finished(client, identifier):
    async with asyncio.timeout(5):
        while True:
            response = await client.get("/v1/runs/" + identifier)
            assert response.status_code == 200, response.text
            run = response.json()
            if run["state"] not in {"queued", "running"}:
                return run
            await asyncio.sleep(0.01)


async def ask(client):
    response = await client.post("/v1/runs", json={"prompt": "Prepare the requested work"})
    assert response.status_code == 202, response.text
    run = await finished(client, response.json()["id"])
    assert run["state"] == "paused", run
    questions = (await client.get("/v1/questions")).json()["questions"]
    assert len(questions) == 1
    return run, questions[0]


@pytest.mark.asyncio
async def test_answers_are_private_partial_persistent_and_resume_exact_tool_result(
    tmp_path, monkeypatch
):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch) as client:
        run, question = await ask(client)
        assert question["resume_run_id"] == run["id"] and "scope" not in question
        assert len(adapter.calls) == 1
        stranger = {"Authorization": "Bearer other-" + 40 * "t"}
        assert (await client.get("/v1/questions", headers=stranger)).json() == {"questions": []}
        denied = await client.post(
            f"/v1/questions/{question['id']}/answer",
            json={"answers": {"q0": ["Report"]}},
            headers=stranger,
        )
        assert denied.status_code == 404 and "Which outputs" not in denied.text
        assert (await client.post(f"/v1/runs/{run['id']}/resume", json={})).status_code == 409
        partial = await client.post(
            f"/v1/questions/{question['id']}/answer",
            json={"answers": {"q0": ["Report", "Custom choice"]}},
        )
        assert partial.status_code == 200, partial.text
        assert partial.json()["status"] == "pending"
        conflict = await client.post(
            f"/v1/questions/{question['id']}/answer", json={"answers": {"q0": ["Chart"]}}
        )
        assert conflict.status_code == 409
        assert len(adapter.calls) == 1
    restarted = service_for(tmp_path, adapter)
    async with client_for(restarted, monkeypatch) as client:
        [saved] = (await client.get("/v1/questions")).json()["questions"]
        assert saved["answers"] == {"q0": ["Report", "Custom choice"]}
        answer = await client.post(
            f"/v1/questions/{question['id']}/answer", json={"answers": {"q1": "Engineering team"}}
        )
        assert answer.status_code == 200 and answer.json()["status"] == "answered", answer.text
        assert len(adapter.calls) == 1
        duplicate = await client.post(
            f"/v1/questions/{question['id']}/answer", json={"answers": {"q1": "Engineering team"}}
        )
        assert duplicate.status_code == 200
        continued = await client.post(f"/v1/runs/{run['id']}/resume", json={})
        assert continued.status_code == 202, continued.text
        assert (await finished(client, continued.json()["id"]))["state"] == "completed"
        assert len(adapter.calls) == 2
        messages = adapter.calls[-1]["messages"]
        replies = [
            message for message in messages if message.role == "tool" and message.name == "clarify"
        ]
        assert len(replies) == 1
        content = json.loads(replies[0].content)
        assert content["responses"][0]["user_response"] == ["Report", "Custom choice"]
        assert content["responses"][1]["user_response"] == "Engineering team"
        assert (await client.get("/v1/questions")).json() == {"questions": []}
        assert (
            await client.post(
                f"/v1/questions/{question['id']}/answer", json={"answers": {"q1": "changed"}}
            )
        ).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["skip", "expire", "cancel-run"])
async def test_question_skip_expiry_and_run_cancellation_never_grant_an_action(
    tmp_path, monkeypatch, action
):
    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch) as client:
        run, question = await ask(client)
        if action == "cancel-run":
            result = await client.post(f"/v1/runs/{run['id']}/cancel", json={})
            assert result.status_code == 200 and result.json()["state"] == "cancelled"
            assert (await client.get("/v1/questions")).json() == {"questions": []}
            assert (
                await client.post(
                    f"/v1/questions/{question['id']}/answer", json={"answers": {"q1": "late"}}
                )
            ).status_code == 409
            assert (await client.post(f"/v1/runs/{run['id']}/resume", json={})).status_code == 409
            assert len(adapter.calls) == 1
            return
        if action == "skip":
            result = await client.post(f"/v1/questions/{question['id']}/cancel", json={})
            assert result.status_code == 200 and result.json()["status"] == "cancelled"
        else:
            assert service.questions.store is not None
            store = service.questions.store
            record = store.get(
                question["id"], scope=service.scope("owner"), session_id=run["session_id"]
            )
            assert record is not None
            record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            with store.db:
                store.db.execute(
                    "UPDATE clarifications SET expires_at=?,payload=? WHERE id=?",
                    (record.expires_at.timestamp(), record.model_dump_json(), record.id),
                )
            [expired] = (await client.get("/v1/questions")).json()["questions"]
            assert expired["status"] == "expired"
            assert (
                await client.post(
                    f"/v1/questions/{question['id']}/answer", json={"answers": {"q1": "late"}}
                )
            ).status_code == 409
        continued = await client.post(f"/v1/runs/{run['id']}/resume", json={})
        assert continued.status_code == 202, continued.text
        assert (await finished(client, continued.json()["id"]))["state"] == "completed"
        reply = next(
            message
            for message in adapter.calls[-1]["messages"]
            if message.role == "tool" and message.name == "clarify"
        )
        assert json.loads(reply.content)["cancelled" if action == "skip" else "timed_out"] is True
        assert await service.storage.list_approvals(status="granted") == []


@pytest.mark.asyncio
async def test_clarify_remains_subject_to_explicit_server_tool_exposure(tmp_path, monkeypatch):
    service = service_for(tmp_path, Adapter(), exposed=())
    async with client_for(service, monkeypatch) as client:
        result = await client.post(
            "/v1/tool-runs",
            json={"name": "clarify", "arguments": {"questions": [{"question": "Ask?"}]}},
        )
        assert result.status_code == 403
        assert (await client.get("/v1/questions")).json() == {"questions": []}


@pytest.mark.asyncio
async def test_unanswered_child_questions_do_not_reserve_more_delegation_budget(
    tmp_path, monkeypatch
):
    from harness.server.delegation import DelegateSubmission

    adapter = Adapter()
    service = service_for(tmp_path, adapter)
    async with client_for(service, monkeypatch) as client:
        await service.delegation.register_parent("owner", "parent", tmp_path)
        child = await service.delegation.submit(
            "owner", "parent", DelegateSubmission(prompt="Prepare a child report", max_steps=3)
        )
        run = await finished(client, child["run"]["id"])
        assert run["state"] == "paused"
        before = await service.delegation.status("owner", child["id"])
        for path in (
            f"/v1/delegations/{child['id']}/resume",
            f"/v1/runs/{run['id']}/resume",
        ):
            blocked = await client.post(path, json={})
            assert blocked.status_code == 409 and "question" in blocked.text, blocked.text
        after = await service.delegation.status("owner", child["id"])
        assert after["budget"] == before["budget"] and after["attempts"] == 1
        [question] = (await client.get("/v1/questions")).json()["questions"]
        answer = await client.post(
            f"/v1/questions/{question['id']}/answer",
            json={"answers": {"q0": ["Report"], "q1": "Team"}},
        )
        assert answer.status_code == 200
        resumed = await client.post(f"/v1/delegations/{child['id']}/resume", json={})
        assert resumed.status_code == 202, resumed.text
        completed = await finished(client, resumed.json()["run"]["id"])
        assert completed["state"] == "completed"
        final = await service.delegation.status("owner", child["id"])
        assert final["attempts"] == 2 and final["budget"]["steps"] == 6
        assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_cancelled_queued_agent_restart_keeps_original_prompt_and_attachment(tmp_path):
    from harness.core import MediaAttachment
    from harness.server import RunSubmission

    from .test_service import FakeAdapter, builder

    original = RunSubmission(
        prompt="Read the attached plan and explain its tradeoffs",
        attachments=[
            MediaAttachment(
                kind="file",
                mime_type="text/plain",
                name="plan.txt",
                data="cGxhbg==",
                model_visible=False,
            )
        ],
    )
    adapter = FakeAdapter()
    service = HarnessService(tmp_path / "queued.db", tmp_path, builder(adapter))
    await service.start(dispatch=False)
    try:
        queued = await service.submit("owner", original)
        cancelled = await service.cancel("owner", queued["id"])
        assert cancelled["state"] == "cancelled" and adapter.calls == 0
        restarted = await service.resume("owner", queued["id"])
        [row] = await service.store.rows(
            "SELECT request FROM api_runs WHERE id=?", (restarted["id"],)
        )
        request = RunSubmission.model_validate_json(row["request"])
        assert request.prompt == original.prompt and request.attachments == original.attachments
    finally:
        await service.close()
    # A process restart executes the retained original request once.
    service = HarnessService(tmp_path / "queued.db", tmp_path, builder(adapter))
    await service.start()
    try:
        async with asyncio.timeout(3):
            async for _ in service.events("owner", restarted["id"]):
                pass
        run = await service.store.run("owner", restarted["id"])
        saved = await service.storage.get(run["session_id"])
        assert run["state"] == "completed" and adapter.calls == 1 and saved is not None
        prompt = next(message for message in saved.messages if message.role == "user")
        assert prompt.content == original.prompt and prompt.attachments == original.attachments
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_question_database_wait_does_not_block_api_writer_commit(tmp_path, monkeypatch):
    from harness.server.questions import QuestionAnswers

    service = service_for(tmp_path, Adapter())
    async with client_for(service, monkeypatch) as client:
        run, question = await ask(client)
        assert service.questions.store is not None
        store = service.questions.store
        owned_record, latest = await service.questions._owned("owner", question["id"])

        async def checked_owner(*args):
            return owned_record, latest

        # Pause after the actual ownership check, so the request reaches the
        # synchronous question writer while another connection holds its lock.
        monkeypatch.setattr(service.questions, "_owned", checked_owner)
        async with service.store.connection() as database:
            await database.execute("BEGIN IMMEDIATE")
            task = asyncio.create_task(
                service.questions.answer(
                    "owner", question["id"], QuestionAnswers(answers={"q0": ["Report"]})
                )
            )
            await asyncio.sleep(0.03)
            assert not task.done()
            await database.commit()
        async with asyncio.timeout(2):
            result = await task
        assert result["answers"] == {"q0": ["Report"]}
        record = store.get(
            question["id"], scope=service.scope("owner"), session_id=run["session_id"]
        )
        assert record is not None and record.status == "pending"


@pytest.mark.asyncio
async def test_answering_does_not_resolve_pending_action_approval(tmp_path, monkeypatch):
    from harness.core import PendingApproval

    service = service_for(tmp_path, Adapter())
    async with client_for(service, monkeypatch) as client:
        run, question = await ask(client)
        approval = await service.storage.create_approval(
            PendingApproval(
                session_id=run["session_id"],
                tool_call_id="separate-action",
                tool_name="write_file",
                arguments={"path": "report.txt"},
            )
        )
        result = await client.post(
            f"/v1/questions/{question['id']}/answer",
            json={"answers": {"q0": ["Report"], "q1": "Team"}},
        )
        assert result.status_code == 200 and result.json()["status"] == "answered"
        saved = await service.storage.get_approval(approval.id)
        assert saved is not None and saved.status == "pending"
        assert (await client.post(f"/v1/runs/{run['id']}/resume", json={})).status_code == 409
        cancelled = await client.post(f"/v1/runs/{run['id']}/cancel", json={})
        assert cancelled.json()["state"] == "cancelled"
        assert service.questions.store is not None
        record = service.questions.store.get(
            question["id"], scope=service.scope("owner"), session_id=run["session_id"]
        )
        assert record is not None and record.status == "cancelled" and not record.applied
        saved = await service.storage.get_approval(approval.id)
        assert saved is not None and saved.status == "denied"


@pytest.mark.asyncio
async def test_mcp_question_operations_use_same_identity_and_answer_validation(
    tmp_path, monkeypatch
):
    service = service_for(tmp_path, Adapter())
    operations = ("questions", "answer_question", "cancel_question")
    async with client_for(
        service, monkeypatch, expose_mcp=True, mcp_operations=operations
    ) as client:
        _, question = await ask(client)
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-11-25",
        }

        async def rpc(method, params, extra=None):
            return await client.post(
                "/mcp/",
                headers={**headers, **(extra or {})},
                json={"jsonrpc": "2.0", "id": "q", "method": method, "params": params},
            )

        initialized = await rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "question-test", "version": "1"},
            },
        )
        assert initialized.status_code == 200
        listing = await rpc("tools/list", {})
        assert {item["name"] for item in listing.json()["result"]["tools"]} == set(operations)
        answer = {
            "name": "answer_question",
            "arguments": {
                "question_id": question["id"],
                "answers": {"q0": ["Report"], "q1": "Team"},
            },
        }
        denied = await rpc("tools/call", answer, {"Authorization": "Bearer other-" + 40 * "t"})
        assert denied.json()["result"]["isError"] and "Which outputs" not in denied.text
        accepted = await rpc("tools/call", answer)
        assert not accepted.json()["result"].get("isError"), accepted.text
        assert accepted.json()["result"]["structuredContent"]["status"] == "answered"
        assert (await client.get("/v1/questions")).json()["questions"][0]["answers"] == {
            "q0": ["Report"],
            "q1": "Team",
        }
