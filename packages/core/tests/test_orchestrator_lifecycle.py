"""Worker failures and cancellation must remain visible and resumable."""

import asyncio
from contextlib import aclosing

import pytest

from harness.core.events import Done, ErrorEvent
from harness.core.orchestrator import (
    AgentRole,
    AgentStartedEvent,
    MultiAgentOrchestrator,
    WorkQueue,
)
from harness.storage.memory import InMemoryStorage


def make_orchestrator(tmp_path, store, factory, **kwargs):
    return MultiAgentOrchestrator(
        agent_factory=factory,
        store=store,
        planner_role=AgentRole(name="planner", system_prompt="plan"),
        worker_role=AgentRole(name="worker", system_prompt="work"),
        reporter_role=AgentRole(name="reporter", system_prompt="report"),
        max_workers=1,
        job_cwd=tmp_path,
        provider="mock",
        model="mock",
        **kwargs,
    )


@pytest.mark.parametrize("failure", ["worker", "factory", "judge", "error_event", "reporter"])
async def test_failures_are_observable_and_root_is_not_successful(tmp_path, failure):
    store = InMemoryStorage()

    class FakeAgent:
        def __init__(self, role):
            self.role = role

        async def run(self, request):
            queue = WorkQueue(store, self.role.job_id)
            if self.role.name == "planner":
                await queue.push("work", cwd=tmp_path)
            elif self.role.name.startswith("worker"):
                if failure == "worker":
                    raise RuntimeError("worker crashed")
                if failure == "error_event":
                    yield ErrorEvent(error="provider failed", kind="network")
                    return
                await queue.complete(self.role.item_id, "done")
            elif failure == "reporter":
                yield ErrorEvent(error="report failed", kind="network")
                return
            yield Done()

    def factory(role):
        if failure == "factory" and role.name.startswith("worker"):
            raise RuntimeError("factory crashed")
        return FakeAgent(role)

    class FailingJudge:
        async def judge(self, **kwargs):
            raise RuntimeError("judge crashed")

    kwargs = {"work_item_judge": FailingJudge()} if failure == "judge" else {}
    events = [
        event
        async for event in make_orchestrator(tmp_path, store, factory, **kwargs).run(
            "work", job_id="job"
        )
    ]
    root = await store.get_task("job")
    assert root is not None and root.status != "done"
    assert any(event.type == "agent_event" and event.event.type == "error" for event in events)
    children = await store.list_tasks(parent_id="job")
    assert all(child.status != "in_progress" for child in children)


async def test_closing_public_stream_cancels_workers_and_requeues_claim(tmp_path):
    store = InMemoryStorage()
    started = asyncio.Event()
    stopped = asyncio.Event()
    mutated = []

    class FakeAgent:
        def __init__(self, role):
            self.role = role

        async def run(self, request):
            if self.role.name == "planner":
                await WorkQueue(store, self.role.job_id).push("work", cwd=tmp_path)
            else:
                started.set()
                try:
                    await asyncio.Event().wait()
                    mutated.append(True)
                finally:
                    stopped.set()
            yield Done()

    orchestrator = make_orchestrator(tmp_path, store, FakeAgent)
    async with aclosing(orchestrator.run("work", job_id="job")) as stream:
        async for event in stream:
            if isinstance(event, AgentStartedEvent) and event.role.startswith("worker"):
                await started.wait()
                break
    assert stopped.is_set()
    assert not mutated
    children = await store.list_tasks(parent_id="job")
    assert [child.status for child in children] == ["todo"]
    root = await store.get_task("job")
    assert root is not None and root.status == "waiting"


async def test_large_queue_cannot_hide_unfinished_item(tmp_path):
    from harness.tasks.schemas import Task

    store = InMemoryStorage()
    await store.create_task(Task(id="large", ref="", title="large", cwd=tmp_path))
    queue = WorkQueue(store, "large")
    await queue.push("unfinished", cwd=tmp_path)
    for index in range(105):
        item = await queue.push(f"finished {index}", cwd=tmp_path)
        await queue.complete(item.id)
    orchestrator = make_orchestrator(tmp_path, store, lambda role: None)
    await orchestrator._finish_job("large")
    root = await store.get_task("large")
    assert root is not None and root.status == "waiting"
    assert len(await queue.list()) == 106


async def test_resume_retries_failed_work_and_can_finish_successfully(tmp_path):
    from harness.tasks.schemas import Task

    store = InMemoryStorage()
    await store.create_task(
        Task(id="resume", ref="", title="resume", cwd=tmp_path, status="waiting")
    )
    queue = WorkQueue(store, "resume")
    item = await queue.push("failed work", cwd=tmp_path)
    await store.update_task(
        item.model_copy(update={"status": "waiting", "metadata": {"_worker_error": "crashed"}})
    )

    class CompletingAgent:
        def __init__(self, role):
            self.role = role

        async def run(self, request):
            if self.role.item_id:
                await queue.complete(self.role.item_id, "retried successfully")
            yield Done()

    orchestrator = make_orchestrator(tmp_path, store, CompletingAgent)
    _events = [event async for event in orchestrator.resume("resume")]
    root = await store.get_task("resume")
    assert root is not None and root.status == "done"
    assert (await queue.list())[0].status == "done"
