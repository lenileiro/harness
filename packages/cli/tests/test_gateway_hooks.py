from __future__ import annotations

from datetime import UTC, datetime

from harness.cli.gateway_hooks import BuiltinHookProvider, WhatsAppNotificationHook
from harness.core.gateway_models import GatewaySessionBinding, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord, ScheduleSpec


def test_builtin_hook_provider_exposes_whatsapp_hook() -> None:
    provider = BuiltinHookProvider()
    hooks = provider.hooks()
    assert len(hooks) == 1
    assert isinstance(hooks[0], WhatsAppNotificationHook)


def test_whatsapp_notification_hook_uses_latest_workspace_session(tmp_path, monkeypatch) -> None:
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session_store.save_session(
        GatewaySessionBinding(
            id="gw-1",
            transport="whatsapp",
            user_id="15550000001",
            thread_id="phone-1",
            updated_at="2026-05-26T12:00:00+00:00",
        )
    )
    session_store.save_session(
        GatewaySessionBinding(
            id="gw-2",
            transport="whatsapp",
            user_id="15550000002",
            thread_id="phone-1",
            updated_at="2026-05-26T12:05:00+00:00",
        )
    )

    captured: dict[str, str] = {}

    def _fake_send(*, to: str, text: str, **kwargs):
        captured["to"] = to
        captured["text"] = text
        return {"messages": [{"id": "wamid.1"}]}

    monkeypatch.delenv("HARNESS_WHATSAPP_NOTIFY_TO", raising=False)
    monkeypatch.setattr("harness.cli.gateway_hooks.send_whatsapp_text_message", _fake_send)

    hook = WhatsAppNotificationHook()
    hook.on_job_completed(
        cwd=tmp_path,
        job=SchedulerJob(
            id="sched-job-1",
            kind="mission.schedule_once",
            cwd=str(tmp_path),
            status="active",
            schedule=ScheduleSpec(kind="at", value="2026-05-26T12:00:00+00:00"),
            next_run_at="2026-05-26T12:00:00+00:00",
        ),
        trigger="scheduled",
        record=SchedulerRunRecord(
            id="schedrun-1",
            job_id="sched-job-1",
            kind="mission.schedule_once",
            cwd=str(tmp_path),
            trigger="scheduled",
            status="completed",
            result_status="completed",
            result_stop_reason="mission_completed",
            started_at=datetime(2026, 5, 26, 12, 0, tzinfo=UTC).isoformat(timespec="seconds"),
            finished_at=datetime(2026, 5, 26, 12, 1, tzinfo=UTC).isoformat(timespec="seconds"),
            record_dir=str(tmp_path / ".harness" / "missions" / "runs" / "schedrun-1"),
            summary="mission.schedule_once -> completed (mission_completed)",
        ),
    )

    assert captured["to"] == "15550000002"
    assert "Harness scheduled run completed." in captured["text"]
    assert "mission.schedule_once" in captured["text"]
