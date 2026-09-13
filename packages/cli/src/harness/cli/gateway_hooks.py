from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from harness.core.approval import PendingApproval
from harness.core.extensions import LifecycleHook
from harness.core.gateway_channels import CHANNEL_LIMITS, ChannelStore
from harness.core.gateway_evidence import approval_expires_at, approval_request_text
from harness.core.gateway_models import GatewayMessage, GatewayReply, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.gateway_whatsapp import send_whatsapp_text_message
from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord


def _latest_whatsapp_user_id(cwd: Path) -> str:
    store = GatewaySessionStore(root=default_gateway_root(cwd))
    sessions = [item for item in store.list_sessions() if item.transport == "whatsapp"]
    if not sessions:
        return ""
    latest = max(
        sessions,
        key=lambda item: (item.updated_at, item.id),
    )
    return latest.user_id


class WhatsAppNotificationHook:
    def prepare_job_notification(self, *, cwd: Path, job: SchedulerJob) -> SchedulerJob:
        """Freeze the destination before the first send so retries keep its recipient."""
        if job.kind in {"reminder.once", "reminder.recurring", "prompt.run"}:
            return job
        return replace(job, payload={**job.payload, "notification_target": self._target(cwd)})

    def _target(self, cwd: Path) -> str:
        explicit = os.environ.get("HARNESS_WHATSAPP_NOTIFY_TO", "").strip()
        if explicit:
            return explicit
        return _latest_whatsapp_user_id(cwd)

    def _send(self, *, cwd: Path, text: str) -> None:
        target = self._target(cwd)
        if not target:
            return
        send_whatsapp_text_message(cwd=cwd, to=target, text=text)

    def on_scheduler_tick(
        self,
        *,
        cwd: Path,
        started_at: datetime,
        finished_at: datetime,
        jobs_seen: int,
        jobs_executed: int,
        run_ids: tuple[str, ...],
    ) -> None:
        return None

    def on_job_started(
        self,
        *,
        cwd: Path,
        job: SchedulerJob,
        trigger: str,
        started_at: datetime,
    ) -> None:
        return None

    def on_job_completed(
        self,
        *,
        cwd: Path,
        job: SchedulerJob,
        trigger: str,
        record: SchedulerRunRecord,
    ) -> None:
        if job.kind == "prompt.run":
            if job.payload.get("transport") != "whatsapp" or not job.payload.get(
                "notify_result", True
            ):
                return
            target = str(job.payload.get("thread_id") or "")
            if target:
                text = str(
                    job.payload.get("notification_text")
                    or f"Scheduled prompt {job.id} failed: {record.summary}"
                )
                send_whatsapp_text_message(cwd=cwd, to=target, text=text)
            return
        if job.kind in {"reminder.once", "reminder.recurring"}:
            if job.payload.get("notify_transport", "whatsapp") != "whatsapp":
                return
            if record.status != "completed":
                return
            target = (
                str(job.payload.get("notify_chat_id", "")).strip()
                or str(job.payload.get("notify_to", "")).strip()
            )
            reminder_text = str(job.payload.get("text", "")).strip() or "Reminder"
            if target:
                send_whatsapp_text_message(
                    cwd=cwd,
                    to=target,
                    text=f"Reminder: {reminder_text}",
                )
            return
        text = (
            "Harness scheduled run completed.\n"
            f"kind: {job.kind}\n"
            f"job: {job.id}\n"
            f"result: {record.result_status} ({record.result_stop_reason})\n"
            f"run: {record.id}"
        )
        if "notification_target" in job.payload:
            target = str(job.payload["notification_target"])
            if target:
                send_whatsapp_text_message(cwd=cwd, to=target, text=text)
        else:
            self._send(cwd=cwd, text=text)

    def on_gateway_message(self, *, cwd: Path, message: GatewayMessage) -> None:
        return None

    def on_gateway_reply(
        self,
        *,
        cwd: Path,
        message: GatewayMessage,
        reply: GatewayReply,
    ) -> None:
        return None

    def on_approval_requested(self, *, cwd: Path, approval: PendingApproval) -> None:
        binding = GatewaySessionStore(root=default_gateway_root(cwd)).load_runtime_binding(
            approval.session_id
        )
        if binding is None or binding.transport != "whatsapp":
            return
        send_whatsapp_text_message(
            cwd=cwd, to=binding.thread_id, text=approval_request_text(approval)
        )

    def on_approval_resolved(self, *, cwd: Path, approval_id: str, granted: bool) -> None:
        return None


class ChannelNotificationHook:
    """Durably hand notifications to the authenticated channel runner's outbox."""

    def prepare_job_notification(self, *, cwd: Path, job: SchedulerJob) -> SchedulerJob:
        return job

    def _queue(
        self,
        *,
        cwd: Path,
        transport: str,
        user_id: str,
        thread_id: str,
        text: str,
        source: str,
        approval: PendingApproval | None = None,
    ) -> None:
        limits = CHANNEL_LIMITS
        if transport not in limits or not user_id or not thread_id:
            return
        store = ChannelStore(cwd=cwd, transport=transport)
        try:
            from harness.cli.channels.approval_interactions import SUPPORTED

            store.queue(
                source=source,
                user_id=user_id,
                thread_id=thread_id,
                text=text,
                limit=limits[transport],
                approval_cards=[
                    {
                        "approval_id": approval.id,
                        "session_id": approval.session_id,
                        "tool_name": approval.tool_name,
                        "expires": approval_expires_at(approval).timestamp(),
                    }
                ]
                if approval is not None and transport in SUPPORTED
                else None,
            )
        finally:
            store.close()

    def on_scheduler_tick(
        self,
        *,
        cwd: Path,
        started_at: datetime,
        finished_at: datetime,
        jobs_seen: int,
        jobs_executed: int,
        run_ids: tuple[str, ...],
    ) -> None:
        return None

    def on_job_started(
        self, *, cwd: Path, job: SchedulerJob, trigger: str, started_at: datetime
    ) -> None:
        return None

    def on_job_completed(
        self, *, cwd: Path, job: SchedulerJob, trigger: str, record: SchedulerRunRecord
    ) -> None:
        if job.kind == "prompt.run" and job.payload.get("notify_result", True):
            transport = str(job.payload.get("transport") or "")
            user_id = str(job.payload.get("user_id") or "")
            thread_id = str(job.payload.get("thread_id") or "")
            text = str(
                job.payload.get("notification_text")
                or f"Scheduled prompt finished: {record.result_status}"
            )
        elif job.kind in {"reminder.once", "reminder.recurring"} and record.status == "completed":
            transport = str(job.payload.get("notify_transport") or "")
            user_id = str(job.payload.get("notify_to") or "")
            thread_id = str(job.payload.get("notify_chat_id") or "")
            text = f"Reminder: {job.payload.get('text') or 'Reminder'}"
        else:
            return
        self._queue(
            cwd=cwd,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
            text=text,
            source=f"scheduler:{record.id}",
        )

    def on_gateway_message(self, *, cwd: Path, message: GatewayMessage) -> None:
        return None

    def on_gateway_reply(self, *, cwd: Path, message: GatewayMessage, reply: GatewayReply) -> None:
        return None

    def on_approval_requested(self, *, cwd: Path, approval: PendingApproval) -> None:
        binding = GatewaySessionStore(root=default_gateway_root(cwd)).load_runtime_binding(
            approval.session_id
        )
        if binding is not None and not binding.local_only:
            self._queue(
                cwd=cwd,
                transport=binding.transport,
                user_id=binding.user_id,
                thread_id=binding.thread_id,
                text=approval_request_text(approval),
                source=f"approval:{approval.id}",
                approval=approval,
            )

    def on_approval_resolved(self, *, cwd: Path, approval_id: str, granted: bool) -> None:
        return None


class BuiltinHookProvider:
    def hooks(self) -> list[LifecycleHook]:
        return [WhatsAppNotificationHook(), ChannelNotificationHook()]


__all__ = [
    "BuiltinHookProvider",
    "ChannelNotificationHook",
    "WhatsAppNotificationHook",
]
