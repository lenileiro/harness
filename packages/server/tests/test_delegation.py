from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from harness.core.errors import NetworkError
from harness.core.schemas import Message, Session, ToolCall
from harness.server import HarnessService, RunSubmission, ServiceError
from harness.server.delegation import (
    DelegateSubmission,
    DelegationLimits,
    DelegationManager,
    LocalDelegationToolset,
    read_job_file,
)

from .test_service import FakeAdapter, builder, client_for, terminal


async def finish(service: HarnessService, run_id: str):
    async with asyncio.timeout(4):
        async for _ in service.events("alice", run_id):
            pass
    return await service.store.run("alice", run_id)


async def test_child_workspaces_inputs_budget_and_artifacts_are_isolated(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "input.txt").write_text("parent content")
    contexts = []
    maxima = []

    class Adapter(FakeAdapter):
        async def stream(self, *, model: str, messages: list[Message], **kwargs: Any):
            maxima.append(kwargs.get("max_tokens"))
            async for event in super().stream(model=model, messages=messages, **kwargs):
                yield event

    base = builder(Adapter())

    def build(context):
        contexts.append(context)
        return base(context)

    service = HarnessService(tmp_path / "state" / "jobs.db", source, build)
    await service.start()
    try:
        await service.delegation.register_parent("alice", "parent", source)
        job = await service.delegation.submit(
            "alice",
            "parent",
            DelegateSubmission(
                prompt="Inspect the provided input",
                inputs=["input.txt"],
                max_steps=3,
                max_tokens=100,
            ),
        )
        assert (await finish(service, job["run"]["id"]))["state"] == "completed"
        assert contexts[0].workspace != source
        assert (contexts[0].workspace / "input.txt").read_text() == "parent content"
        assert maxima == [100]
        (contexts[0].workspace / "result.txt").write_text("child result")
        artifact = await service.delegation.artifact("alice", job["id"], "result.txt", "parent")
        assert base64.b64decode(artifact.data or "") == b"child result"
        assert artifact.model_visible is False
        assert not (source / "result.txt").exists()
        status = await service.delegation.status("alice", job["id"], "parent")
        assert status["budget"]["steps"] == 3 and status["budget"]["output_tokens"] == 300
        assert "result.txt" in status["artifacts"] and status["summary"]
        for who, parent in (("bob", "parent"), ("alice", "sibling")):
            with pytest.raises(ServiceError, match="not found"):
                await service.delegation.status(who, job["id"], parent)
        with pytest.raises(ServiceError, match="reserve"):
            await service.submit(
                "alice", RunSubmission(prompt="bypass budget", session_id=job["session_id"])
            )
        resumed = await service.resume("alice", job["run"]["id"], "Explicit follow up")
        await finish(service, resumed["id"])
        assert (await service.delegation.status("alice", job["id"]))["budget"]["steps"] == 6
    finally:
        await service.close()


async def test_budget_reservations_are_atomic_and_survive_restart(tmp_path):
    service = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    service.delegation = DelegationManager(service, DelegationLimits(max_children=1))
    await service.start(dispatch=False)
    await service.delegation.register_parent("alice", "parent", tmp_path)
    results = await asyncio.gather(
        *(
            service.delegation.submit("alice", "parent", DelegateSubmission(prompt="one child"))
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    jobs = [result for result in results if isinstance(result, dict)]
    assert len(jobs) == 1 and sum(isinstance(result, ServiceError) for result in results) == 1
    assert len(list((tmp_path / "jobs").iterdir())) == 1
    await service.close()
    restarted = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    await restarted.start(dispatch=False)
    try:
        status = await restarted.delegation.status("alice", jobs[0]["id"])
        assert status["state"] == "queued" and status["budget"]["children"] == 1
        assert status["budget"]["limits"]["max_children"] == 1
        with pytest.raises(ServiceError, match="budget exhausted"):
            await restarted.delegation.submit(
                "alice", "parent", DelegateSubmission(prompt="another child")
            )
    finally:
        await restarted.close()


async def test_depth_and_descendant_cancellation_share_root_budget(tmp_path):
    service = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    await service.start(dispatch=False)
    try:
        await service.delegation.register_parent("alice", "parent", tmp_path)
        child = await service.delegation.submit(
            "alice", "parent", DelegateSubmission(prompt="child")
        )
        grandchild = await service.delegation.submit(
            "alice", child["session_id"], DelegateSubmission(prompt="grandchild")
        )
        with pytest.raises(ServiceError, match="depth budget"):
            await service.delegation.submit(
                "alice", grandchild["session_id"], DelegateSubmission(prompt="too deep")
            )
        await service.delegation.cancel("alice", child["id"], "parent")
        assert (await service.delegation.status("alice", child["id"]))["state"] == "cancelled"
        assert (await service.delegation.status("alice", grandchild["id"]))["state"] == "cancelled"
        assert (await service.delegation.status("alice", child["id"]))["budget"]["children"] == 2
    finally:
        await service.close()


async def test_timeout_covers_builder_and_model_and_never_reports_success(tmp_path):
    async def blocked(context):
        await asyncio.Event().wait()
        raise AssertionError

    service = HarnessService(tmp_path / "jobs.db", tmp_path, blocked)
    await service.start()
    try:
        await service.delegation.register_parent("alice", "parent", tmp_path)
        job = await service.delegation.submit(
            "alice", "parent", DelegateSubmission(prompt="timeout", timeout_seconds=1)
        )
        run = await finish(service, job["run"]["id"])
        assert run["state"] == "failed" and "elapsed-time budget" in run["error"]
    finally:
        await service.close()


async def test_allocation_counts_failover_attempts_before_provider_call(tmp_path):
    class FailingAdapter(FakeAdapter):
        async def stream(self, *, model: str, messages: list[Message], **kwargs: Any):
            self.calls += 1
            raise NetworkError("simulated offline error")
            yield  # pragma: no cover

    adapter = FailingAdapter()
    base = builder(adapter)

    def build(context):
        agent = base(context)
        agent.failover.max_attempts = 3
        agent.failover.backoff_base = 0
        return agent

    service = HarnessService(tmp_path / "jobs.db", tmp_path, build)
    await service.start()
    try:
        await service.delegation.register_parent("alice", "parent", tmp_path)
        job = await service.delegation.submit(
            "alice", "parent", DelegateSubmission(prompt="budget", max_steps=1)
        )
        assert (await finish(service, job["run"]["id"]))["state"] == "failed"
        assert adapter.calls == 1
    finally:
        await service.close()


def test_input_copy_rejects_symlinks_secrets_parent_traversal_and_special_files(tmp_path):
    (tmp_path / "ordinary.txt").write_text("safe")
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "link.txt").symlink_to(tmp_path / "ordinary.txt")
    assert read_job_file(tmp_path, "ordinary.txt") == b"safe"
    for path in (
        "../ordinary.txt",
        ".env",
        ".harness/state.db",
        "linked/ordinary.txt",
        "link.txt",
        str(tmp_path / "ordinary.txt"),
    ):
        with pytest.raises(ServiceError):
            read_job_file(tmp_path, path)
    import os

    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "fifo")
        with pytest.raises(ServiceError):
            read_job_file(tmp_path, "fifo")


async def test_model_delegation_rpc_uses_existing_approval_and_agent_pipeline(
    tmp_path, monkeypatch
):
    service = HarnessService(
        tmp_path / "jobs.db",
        tmp_path,
        builder(FakeAdapter()),
        exposed_tools=["delegate", "delegate_status"],
    )
    async with client_for(service, monkeypatch) as client:
        run = (
            await client.post(
                "/v1/tool-runs", json={"name": "delegate", "arguments": {"prompt": "child task"}}
            )
        ).json()
        assert (await terminal(client, run["id"]))["state"] == "paused"
        assert await service.delegation.list("alice") == []
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        resumed = (await client.post(f"/v1/runs/{run['id']}/resume", json={})).json()
        assert (await terminal(client, resumed["id"]))["state"] == "completed"
        jobs = await service.delegation.list("alice")
        assert len(jobs) == 1
        assert (await terminal(client, jobs[0]["run"]["id"]))["state"] == "completed"


async def test_local_toolset_binds_parent_without_changing_its_scope(tmp_path):
    parent = Session(id="local-parent", provider="fake", model="test", cwd=tmp_path)
    async with LocalDelegationToolset(
        tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()), owner="alice"
    ) as toolset:
        tools = await toolset.bind(parent)
        tool = next(tool for tool in tools if tool.name == "delegate")
        result = await tool(
            ToolCall(id="spawn", name="delegate", arguments={"prompt": "local child"})
        )
        assert not result.is_error
    assert parent.metadata == {}
    service = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    await service.start(dispatch=False)
    try:
        assert (await service.delegation.list("alice", parent.id))[0]["state"] == "completed"
    finally:
        await service.close()


async def test_administrative_status_and_cancel_work_while_worker_owns_lease(tmp_path):
    adapter = FakeAdapter(block=asyncio.Event())
    worker = HarnessService(tmp_path / "jobs.db", tmp_path, builder(adapter))
    await worker.start()
    admin = HarnessService(
        tmp_path / "jobs.db",
        tmp_path,
        lambda context: pytest.fail("admin status must not construct an adapter"),
    )
    try:
        await worker.delegation.register_parent("alice", "parent", tmp_path)
        job = await worker.delegation.submit(
            "alice", "parent", DelegateSubmission(prompt="running job")
        )
        await asyncio.wait_for(adapter.entered.wait(), 2)
        await admin.start(dispatch=False)
        assert (await admin.delegation.status("alice", job["id"]))["state"] == "running"
        requested = await admin.delegation.cancel("alice", job["id"])
        assert requested["run"]["cancellation_requested"] is True
        assert (await finish(admin, job["run"]["id"]))["state"] == "cancelled"
        assert adapter.cancelled.is_set()
    finally:
        await admin.close()
        await worker.close()


async def test_cancelled_commit_keeps_durable_job_inputs(tmp_path, monkeypatch):
    (tmp_path / "input.txt").write_text("preserve this input")
    service = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    await service.start(dispatch=False)
    try:
        await service.delegation.register_parent("alice", "parent", tmp_path)
        connection = service.store.db
        assert connection is not None
        original = connection.commit

        async def committed_then_cancelled():
            await original()
            raise asyncio.CancelledError

        monkeypatch.setattr(connection, "commit", committed_then_cancelled)
        with pytest.raises(asyncio.CancelledError):
            await service.delegation.submit(
                "alice", "parent", DelegateSubmission(prompt="persist", inputs=["input.txt"])
            )
        monkeypatch.setattr(connection, "commit", original)
        jobs = await service.delegation.list("alice")
        assert len(jobs) == 1 and jobs[0]["state"] == "queued"
        artifact = await service.delegation.artifact("alice", jobs[0]["id"], "input.txt")
        assert base64.b64decode(artifact.data or "") == b"preserve this input"
    finally:
        await service.close()


async def test_http_child_handles_scope_and_artifact_download(tmp_path, monkeypatch):
    (tmp_path / "input.txt").write_text("source input")
    service = HarnessService(tmp_path / "jobs.db", tmp_path, builder(FakeAdapter()))
    async with client_for(service, monkeypatch) as client:
        parent = (await client.post("/v1/runs", json={"prompt": "parent task"})).json()
        await terminal(client, parent["id"])
        response = await client.post(
            "/v1/delegations",
            json={
                "parent_session_id": parent["session_id"],
                "prompt": "uniquely searchable child",
                "inputs": ["input.txt"],
            },
        )
        assert response.status_code == 202, response.text
        job = response.json()
        child = await terminal(client, job["run"]["id"])
        assert child["delegation_handle"] == job["id"]
        assert (await client.get("/v1/sessions?q=uniquely")).json()["sessions"][0][
            "session_id"
        ] == job["session_id"]
        artifact = await client.get(f"/v1/delegations/{job['id']}/artifacts?path=input.txt")
        assert artifact.content == b"source input"
        from .test_service import bob_headers

        for path in (
            f"/v1/delegations/{job['id']}",
            f"/v1/delegations/{job['id']}/artifacts?path=input.txt",
        ):
            assert (await client.get(path, headers=bob_headers())).status_code == 404
        assert (await client.get("/v1/delegations", headers=bob_headers())).json()["jobs"] == []


async def test_standalone_agent_dispatches_bound_delegation_tools(tmp_path):
    from harness.core import Agent
    from harness.core.events import Done, ToolCallEvent
    from harness.core.failover import FailoverPolicy
    from harness.core.schemas import RunRequest
    from harness.core.tools import ToolRegistry
    from harness.storage.sqlite import SQLiteStorage

    class ParentAdapter(FakeAdapter):
        async def stream(self, *, model: str, messages: list[Message], **kwargs: Any):
            self.calls += 1
            if self.calls == 1:
                assert any(schema["function"]["name"] == "delegate" for schema in kwargs["tools"])
                call = ToolCall(
                    id="child-call", name="delegate", arguments={"prompt": "standalone child"}
                )
                yield ToolCallEvent(call=call)
                yield Done(final_message=Message(role="assistant", tool_calls=[call]))
            else:
                yield Done(final_message=Message(role="assistant", content="Child is queued"))

    storage = SQLiteStorage(path=tmp_path / "parent.db")
    try:
        async with LocalDelegationToolset(
            tmp_path / "children.db", tmp_path, builder(FakeAdapter()), owner="alice"
        ) as children:
            agent = Agent(
                adapters={"fake": ParentAdapter()},
                tools=ToolRegistry(),
                storage=storage,
                failover=FailoverPolicy(chain=["fake"]),
                default_model="test",
                default_cwd=str(tmp_path),
                session_tool_factory=children.bind,
            )
            async for _ in agent.run(
                RunRequest(prompt="Delegate independent work", session_id="parent")
            ):
                pass
        saved = await storage.get("parent")
        from harness.core.session_search import session_scope

        assert saved is not None
        scope = session_scope(saved)
        assert scope is not None and scope.user_id is None
        assert any(message.tool_call_id == "child-call" for message in saved.messages)
    finally:
        await storage.close()
