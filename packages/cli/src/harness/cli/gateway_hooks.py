from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from harness.core.extensions import LifecycleHook
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
        if job.kind in {"reminder.once", "reminder.recurring"}:
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
        self._send(
            cwd=cwd,
            text=(
                "Harness scheduled run completed.\n"
                f"kind: {job.kind}\n"
                f"job: {job.id}\n"
                f"result: {record.result_status} ({record.result_stop_reason})\n"
                f"run: {record.id}"
            ),
        )

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

    def on_approval_requested(self, *, cwd: Path, approval) -> None:
        return None

    def on_approval_resolved(self, *, cwd: Path, approval_id: str, granted: bool) -> None:
        return None


class BuiltinHookProvider:
    def hooks(self) -> list[LifecycleHook]:
        return [WhatsAppNotificationHook()]


__all__ = [
    "BuiltinHookProvider",
    "WhatsAppNotificationHook",
]
