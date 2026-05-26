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
