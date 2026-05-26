from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from harness.core.approval import ApprovalStore
from harness.core.extensions import LifecycleHook
from harness.core.gateway_models import GatewayMessage, GatewayReply, GatewaySessionBinding
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.mission_reporter import build_mission_summary_report
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.research_store import ResearchStore, default_research_root
from harness.core.scheduler_runtime import (
    create_scheduler_job,
    parse_schedule_spec,
    run_scheduler_job,
)
from harness.core.scheduler_store import SchedulerStore
from harness.core.shared_queue import build_shared_work_queue


def _utcnow_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tokenize(text: str) -> list[str]:
    return [part.strip() for part in text.strip().split() if part.strip()]


async def _pending_approval_count(store: ApprovalStore | None) -> int:
    if store is None:
        return 0
    return len(await store.list_approvals(status="pending"))


async def _resolve_approval(
    approval_store: ApprovalStore | None, approval_id: str
) -> tuple[bool, str]:
    if approval_store is None:
        return False, "Approval store is not configured."
    updated = await approval_store.resolve_approval(
        approval_id, status="granted", resolved_by="gateway"
    )
    if updated is None:
        return False, f"Approval not found: {approval_id}"
    return True, f"Granted {updated.id} ({updated.tool_name})."


def _latest_runs(store: SchedulerStore, *, limit: int = 5) -> list[dict[str, str]]:
    runs = sorted(
        store.list_run_records(),
        key=lambda item: (item.finished_at, item.id),
        reverse=True,
    )
    return [
        {
            "id": item.id,
            "job_id": item.job_id,
            "kind": item.kind,
            "status": item.status,
            "result_status": item.result_status,
            "finished_at": item.finished_at,
        }
        for item in runs[:limit]
    ]


def _format_runs(runs: list[dict[str, str]]) -> str:
    if not runs:
        return "No scheduler runs recorded."
    lines = ["Latest runs:"]
    for item in runs:
        lines.append(
            f"- {item['id']} {item['kind']} -> {item['result_status']} " f"({item['finished_at']})"
        )
    return "\n".join(lines)


def _dispatch_mission_start(
    *,
    cwd: Path,
    scheduler_store: SchedulerStore,
    mission_id: str,
) -> tuple[dict[str, str], str]:
    job = create_scheduler_job(
        store=scheduler_store,
        kind="mission.schedule_once",
        cwd=cwd,
        schedule=parse_schedule_spec(at=_utcnow_text()),
        payload={"mission_id": mission_id, "max_steps": 20, "auto_complete": False},
        title=mission_id,
    )
    scheduler_store.add_job(job)
    record = run_scheduler_job(store=scheduler_store, job_id=job.id, trigger="gateway")
    return {
        "job_id": job.id,
        "run_id": record.id,
        "mission_id": mission_id,
        "result_status": record.result_status,
        "record_dir": record.record_dir,
    }, f"Started mission {mission_id}: {record.result_status} ({record.result_stop_reason})."


def _dispatch_research_start(
    *,
    cwd: Path,
    scheduler_store: SchedulerStore,
) -> tuple[dict[str, str], str]:
    job = create_scheduler_job(
        store=scheduler_store,
        kind="research.schedule_once",
        cwd=cwd,
        schedule=parse_schedule_spec(at=_utcnow_text()),
        payload={
            "max_steps": 5,
            "max_risk": "medium",
            "base_branch": "main",
            "create_branch": False,
            "commit": False,
            "push": False,
            "open_pr": False,
            "draft_pr": True,
        },
        title=f"research-{cwd.name}",
    )
    scheduler_store.add_job(job)
    record = run_scheduler_job(store=scheduler_store, job_id=job.id, trigger="gateway")
    return {
        "job_id": job.id,
        "run_id": record.id,
        "result_status": record.result_status,
        "record_dir": record.record_dir,
    }, f"Started research burst: {record.result_status} ({record.result_stop_reason})."


async def dispatch_gateway_message(
    *,
    cwd: Path,
    session_store: GatewaySessionStore,
    scheduler_store: SchedulerStore,
    message: GatewayMessage,
    approval_store: ApprovalStore | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
) -> tuple[GatewayReply, GatewaySessionBinding]:
    for hook in hooks:
        hook.on_gateway_message(cwd=cwd, message=message)
    session = session_store.get_or_create_session(
        transport=message.transport,
        user_id=message.user_id,
        thread_id=message.thread_id,
    )
    tokens = _tokenize(message.text)
    if not tokens:
        return (
            GatewayReply(
                session_id=session.id,
                command="empty",
                status="error",
                text="Empty gateway message.",
            ),
            session,
        )

    lowered = [item.lower() for item in tokens]
    command = lowered[0]
    mission_store = MissionStore(root=default_mission_root(cwd))

    if command == "status":
        jobs = scheduler_store.list_jobs()
        active_jobs = sum(1 for item in jobs if item.status == "active")
        paused_jobs = sum(1 for item in jobs if item.status == "paused")
        mission_count = len(mission_store.list_missions())
        research_store = ResearchStore(root=default_research_root(cwd))
        shared_queue = build_shared_work_queue(
            mission_store=mission_store,
            research_store=research_store,
        )
        ready_queue = [item for item in shared_queue if item.ready]
        pending_approvals = await _pending_approval_count(approval_store)
        reply = GatewayReply(
            session_id=session.id,
            command="status",
            status="ok",
            text=(
                f"Scheduler jobs: {len(jobs)} total, {active_jobs} active, {paused_jobs} paused. "
                f"Missions: {mission_count}. Shared queue: {len(shared_queue)} item(s), "
                f"{len(ready_queue)} ready. Pending approvals: {pending_approvals}."
            ),
            data={
                "jobs_total": len(jobs),
                "jobs_active": active_jobs,
                "jobs_paused": paused_jobs,
                "missions_total": mission_count,
                "shared_queue_total": len(shared_queue),
                "shared_queue_ready": len(ready_queue),
                "shared_queue_top": shared_queue[0].to_dict() if shared_queue else None,
                "pending_approvals": pending_approvals,
            },
        )
    elif command == "runs":
        runs = _latest_runs(scheduler_store)
        reply = GatewayReply(
            session_id=session.id,
            command="runs",
            status="ok",
            text=_format_runs(runs),
            data={"runs": runs},
        )
    elif command == "mission" and len(tokens) >= 3 and lowered[1] == "start":
        mission_id = tokens[2]
        payload, text = _dispatch_mission_start(
            cwd=cwd,
            scheduler_store=scheduler_store,
            mission_id=mission_id,
        )
        session = replace(
            session,
            current_mission_id=mission_id,
            last_job_id=payload["job_id"],
            last_run_id=payload["run_id"],
            last_command="mission start",
            updated_at=_utcnow_text(),
        )
        session_store.save_session(session)
        reply = GatewayReply(
            session_id=session.id,
            command="mission.start",
            status="ok",
            text=text,
            data=payload,
        )
    elif command == "research" and len(tokens) >= 2 and lowered[1] == "start":
        payload, text = _dispatch_research_start(cwd=cwd, scheduler_store=scheduler_store)
        session = replace(
            session,
            last_job_id=payload["job_id"],
            last_run_id=payload["run_id"],
            last_command="research start",
            updated_at=_utcnow_text(),
        )
        session_store.save_session(session)
        reply = GatewayReply(
            session_id=session.id,
            command="research.start",
            status="ok",
            text=text,
            data=payload,
        )
    elif command == "approve" and len(tokens) >= 2:
        ok, text = await _resolve_approval(approval_store, tokens[1])
        session = replace(session, last_command="approve", updated_at=_utcnow_text())
        session_store.save_session(session)
        for hook in hooks:
            hook.on_approval_resolved(cwd=cwd, approval_id=tokens[1], granted=ok)
        reply = GatewayReply(
            session_id=session.id,
            command="approve",
            status="ok" if ok else "error",
            text=text,
            data={"approval_id": tokens[1]},
        )
    elif command == "report" and len(tokens) >= 2:
        mission_id = tokens[1]
        report = build_mission_summary_report(store=mission_store, mission_id=mission_id)
        session = replace(
            session,
            current_mission_id=mission_id,
            last_command="report",
            updated_at=_utcnow_text(),
        )
        session_store.save_session(session)
        text = report.summary
        if report.next_actions:
            text += "\nNext: " + "; ".join(report.next_actions)
        reply = GatewayReply(
            session_id=session.id,
            command="report",
            status="ok",
            text=text,
            data=report.to_dict(),
        )
    else:
        reply = GatewayReply(
            session_id=session.id,
            command="unknown",
            status="error",
            text=(
                "Unsupported gateway command. Use one of: status, runs, "
                "mission start <id>, research start, approve <id>, report <mission_id>."
            ),
        )
    if reply.command not in {"mission.start", "research.start", "approve", "report"}:
        session = replace(session, last_command=reply.command, updated_at=_utcnow_text())
        session_store.save_session(session)
    for hook in hooks:
        hook.on_gateway_reply(cwd=cwd, message=message, reply=reply)
    return reply, session


__all__ = ["dispatch_gateway_message"]
