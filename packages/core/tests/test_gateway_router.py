from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from harness.core.approval import ApprovalStore
from harness.core.gateway_models import GatewayMessage
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler_store import SchedulerStore


def test_gateway_router_emits_hooks_for_message_reply_and_approval(tmp_path: Path) -> None:
    class RecordingHook:
        def __init__(self) -> None:
            self.events: list[tuple[object, ...]] = []

        def on_scheduler_tick(self, **kwargs) -> None:
            self.events.append(("tick", kwargs))

        def on_job_started(self, **kwargs) -> None:
            self.events.append(("started", kwargs))

        def on_job_completed(self, **kwargs) -> None:
            self.events.append(("completed", kwargs))

        def on_gateway_message(self, **kwargs) -> None:
            self.events.append(("message", kwargs["message"].text))

        def on_gateway_reply(self, **kwargs) -> None:
            self.events.append(("reply", kwargs["reply"].command))

        def on_approval_requested(self, **kwargs) -> None:
            self.events.append(("approval_requested", kwargs))

        def on_approval_resolved(self, **kwargs) -> None:
            self.events.append(("approval_resolved", kwargs["approval_id"], kwargs["granted"]))

    class ApprovalStoreStub:
        async def resolve_approval(self, approval_id: str, *, status: str, resolved_by: str):
            class Approval:
                id = approval_id
                tool_name = "shell"

            return Approval()

    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    hook = RecordingHook()

    async def _run() -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id="msg-1",
                transport="local",
                user_id="u1",
                thread_id="t1",
                text="approve appr_123",
            ),
            approval_store=cast(ApprovalStore, ApprovalStoreStub()),
            hooks=(hook,),
        )
        assert reply.command == "approve"
        assert reply.status == "ok"

    asyncio.run(_run())

    assert hook.events[0] == ("message", "approve appr_123")
    assert hook.events[1] == ("approval_resolved", "appr_123", True)
    assert hook.events[2] == ("reply", "approve")


def test_gateway_router_can_schedule_reminder_intent(tmp_path: Path, monkeypatch) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    launched: dict[str, object] = {}

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            launched["args"] = list(args)
            launched["cwd"] = kwargs.get("cwd")

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)

    async def _run() -> None:
        reply, session = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id="msg-1",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text="remind me in 5 minutes to check the build",
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"
        assert "I'll remind you in 5 minute(s)" in reply.text
        assert session.last_job_id

    asyncio.run(_run())

    jobs = scheduler_store.list_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.kind == "reminder.once"
    assert job.payload["text"] == "check the build"
    assert job.payload["notify_to"] == "15551234567"
    assert job.payload["notify_chat_id"] == "15551234567@s.whatsapp.net"
    assert cast(list[str], launched["args"])[1:4] == ["run", "harness", "scheduler"]
    profile = session_store.get_or_create_profile(transport="whatsapp", user_id="15551234567")
    assert profile.recent_threads == ["15551234567@s.whatsapp.net"]
    assert len(profile.active_work) == 1
    assert profile.active_work[0].ref == f"job:{job.id}"
    assert profile.active_work[0].summary.startswith("Okay. I'll remind you")


def test_gateway_router_can_schedule_daily_reminder(tmp_path: Path, monkeypatch) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    launched: dict[str, object] = {}

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            launched["args"] = list(args)

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)

    async def _run() -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id="msg-2",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text="remind me daily to stand up",
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"
        assert reply.text == "Okay. I'll remind you every day: stand up"

    asyncio.run(_run())

    job = scheduler_store.list_jobs()[0]
    assert job.kind == "reminder.recurring"
    assert job.schedule.kind == "cron"
    assert job.schedule.value.count(" ") == 4
    assert "--max-ticks" not in cast(list[str], launched["args"])


def test_gateway_router_reuses_existing_scheduler_watcher(tmp_path: Path, monkeypatch) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    launches: list[list[str]] = []

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            self.pid = 4242
            launches.append(list(args))

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)
    monkeypatch.setattr("harness.core.gateway_router.os.kill", lambda pid, sig: None)

    async def _run(text: str) -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id=f"msg-{text}",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text=text,
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"

    asyncio.run(_run("remind me in 2 minutes to check the deploy"))
    asyncio.run(_run("remind me in 1 minute to check the build"))

    assert len(launches) == 1


def test_gateway_router_launches_watcher_when_existing_one_expires_too_soon(
    tmp_path: Path, monkeypatch
) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    launches: list[list[str]] = []

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            self.pid = 4242 + len(launches)
            launches.append(list(args))

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)
    monkeypatch.setattr("harness.core.gateway_router.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("harness.core.gateway_router.time.time", lambda: 1000.0)

    async def _run(text: str) -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id=f"msg-{text}",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text=text,
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"

    asyncio.run(_run("remind me in 1 minute to check the build"))
    asyncio.run(_run("remind me in 2 hours to check the deploy"))

    assert len(launches) == 2


def test_gateway_router_can_schedule_weekday_reminder(tmp_path: Path, monkeypatch) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            return None

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)

    async def _run() -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id="msg-3",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text="remind me every tuesday to send the report",
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"
        assert "every tuesday" in reply.text

    asyncio.run(_run())

    job = scheduler_store.list_jobs()[0]
    assert job.kind == "reminder.recurring"
    assert job.schedule.kind == "cron"
    assert job.schedule.value.endswith(" 2")


def test_gateway_router_can_schedule_weekly_and_monthly_reminders(
    tmp_path: Path, monkeypatch
) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")

    class PopenStub:
        def __init__(self, args, **kwargs) -> None:
            return None

    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", PopenStub)

    async def _run(text: str) -> None:
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id=f"msg-{text}",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                text=text,
            ),
        )
        assert reply.command == "reminder.create"
        assert reply.status == "ok"

    asyncio.run(_run("remind me weekly to review metrics"))
    asyncio.run(_run("remind me monthly to close the books"))

    jobs = scheduler_store.list_jobs()
    assert len(jobs) == 2
    schedules_by_text = {str(job.payload["text"]): job.schedule for job in jobs}
    weekly_schedule = schedules_by_text["review metrics"]
    monthly_schedule = schedules_by_text["close the books"]
    assert weekly_schedule.kind == "cron"
    assert monthly_schedule.kind == "cron"
    assert weekly_schedule.value.split()[-1].isdigit()
    assert monthly_schedule.value.split()[2].isdigit()
