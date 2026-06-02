from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.core.approval import ApprovalStore
from harness.core.dynamic_workflows import (
    WorkflowStore,
    create_default_workflow,
    default_workflow_root,
    render_workflow_mermaid,
)
from harness.core.extensions import LifecycleHook
from harness.core.gateway_models import (
    GatewayMessage,
    GatewayReply,
    GatewaySessionBinding,
    GatewayWorkRef,
)
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


def is_gateway_control_message(text: str) -> bool:
    tokens = _tokenize(text)
    if not tokens:
        return True
    lowered = [item.lower() for item in tokens]
    command = lowered[0]
    if command in {"status", "runs"}:
        return True
    if command == "mission" and len(tokens) >= 3 and lowered[1] == "start":
        return True
    if command == "research" and len(tokens) >= 2 and lowered[1] == "start":
        return True
    if command == "workflow" and len(tokens) >= 2:
        workflow_commands = {"list", "start", "status", "resume", "report", "cancel", "events"}
        if lowered[1] in workflow_commands:
            return True
        if lowered[1] == "graph":
            return True
    if command in {"approve", "report"} and len(tokens) >= 2:
        return True
    return _parse_reminder_intent(text) is not None


def _weekday_to_cron(value: str) -> str | None:
    mapping = {
        "sunday": "0",
        "monday": "1",
        "tuesday": "2",
        "wednesday": "3",
        "thursday": "4",
        "friday": "5",
        "saturday": "6",
    }
    return mapping.get(value.strip().lower())


def _current_time_cron(*, now: datetime) -> tuple[int, int]:
    current = now.astimezone(UTC)
    return current.minute, current.hour


def _parse_reminder_intent(text: str) -> tuple[str, str, str, str] | None:
    normalized = " ".join(text.strip().split())
    patterns = (
        r"^remind me in (?P<amount>\d+) (?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)"
        r"(?: to| about)? (?P<body>.+)$",
        r"^in (?P<amount>\d+) (?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?),? "
        r"remind me(?: to| about)? (?P<body>.+)$",
    )
    match = None
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            break
    if match is None:
        recurring_patterns = (
            r"^remind me (?P<freq>daily|weekly|monthly)(?: to| about)? (?P<body>.+)$",
            r"^remind me every (?P<freq>day|week|month)(?: to| about)? (?P<body>.+)$",
            r"^remind me every (?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            r"(?: to| about)? (?P<body>.+)$",
            r"^every (?P<weekday2>monday|tuesday|wednesday|thursday|friday|saturday|sunday),? "
            r"remind me(?: to| about)? (?P<body2>.+)$",
        )
        recurring_match = None
        for pattern in recurring_patterns:
            recurring_match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if recurring_match:
                break
        if recurring_match is None:
            return None
        body = (
            (
                recurring_match.groupdict().get("body")
                or recurring_match.groupdict().get("body2")
                or ""
            )
            .strip()
            .rstrip(".")
        )
        if not body:
            return None
        now = datetime.now(UTC)
        minute, hour = _current_time_cron(now=now)
        weekday_name = (
            recurring_match.groupdict().get("weekday")
            or recurring_match.groupdict().get("weekday2")
            or ""
        ).strip()
        if weekday_name:
            weekday = _weekday_to_cron(weekday_name)
            if weekday is None:
                return None
            return (
                "reminder.recurring",
                f"{minute} {hour} * * {weekday}",
                body,
                f"every {weekday_name.lower()}",
            )
        freq = str(recurring_match.groupdict().get("freq", "")).strip().lower()
        if freq in {"daily", "day"}:
            return ("reminder.recurring", f"{minute} {hour} * * *", body, "every day")
        if freq in {"weekly", "week"}:
            weekday = str((now.weekday() + 1) % 7)
            return ("reminder.recurring", f"{minute} {hour} * * {weekday}", body, "every week")
        if freq in {"monthly", "month"}:
            day_of_month = now.day
            return (
                "reminder.recurring",
                f"{minute} {hour} {day_of_month} * *",
                body,
                "every month",
            )
        return None
    amount = int(match.group("amount"))
    unit = match.group("unit").lower()
    body = match.group("body").strip().rstrip(".")
    if amount < 1 or not body:
        return None
    if unit.startswith(("second", "sec")):
        multiplier = 1
    elif unit.startswith(("minute", "min")):
        multiplier = 60
    else:
        multiplier = 3600
    wait_seconds = amount * multiplier
    if unit.startswith(("second", "sec")):
        label = f"in {wait_seconds} second(s)"
    elif unit.startswith(("minute", "min")):
        label = f"in {amount} minute(s)"
    else:
        label = f"in {amount} hour(s)"
    return ("reminder.once", str(wait_seconds), body, label)


def _launch_scheduler_watcher(*, cwd: Path, wait_seconds: int, recurring: bool = False) -> None:
    watcher_root = cwd / ".harness" / "gateway"
    watcher_root.mkdir(parents=True, exist_ok=True)
    watcher_pid_path = watcher_root / "scheduler-watcher.pid"
    if watcher_pid_path.is_file():
        try:
            raw_record = watcher_pid_path.read_text(encoding="utf-8").strip()
            try:
                record = json.loads(raw_record)
            except json.JSONDecodeError:
                record = {"pid": int(raw_record)}
            existing_pid = int(record["pid"])
            os.kill(existing_pid, 0)
            existing_recurring = bool(record.get("recurring", False))
            existing_expires_at = record.get("expires_at")
            existing_covers_request = existing_recurring or (
                not recurring
                and isinstance(existing_expires_at, int | float)
                and float(existing_expires_at) >= time.time() + wait_seconds
            )
            if existing_covers_request:
                return
        except (OSError, TypeError, ValueError, KeyError):
            watcher_pid_path.unlink(missing_ok=True)
    uv_bin = shutil.which("uv") or "uv"
    poll_interval = 5.0 if wait_seconds >= 30 else 1.0
    command = [
        uv_bin,
        "run",
        "harness",
        "scheduler",
        "start",
        "--cwd",
        str(cwd),
        "--poll-interval",
        str(poll_interval),
    ]
    max_ticks: int | None = None
    if not recurring:
        max_ticks = max(6, int(wait_seconds / poll_interval) + 12)
        command.extend(["--max-ticks", str(max_ticks)])
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=os.environ.copy(),
    )
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        expires_at = (
            None if recurring or max_ticks is None else time.time() + max_ticks * poll_interval
        )
        watcher_pid_path.write_text(
            json.dumps({"pid": pid, "recurring": recurring, "expires_at": expires_at}),
            encoding="utf-8",
        )


def _dispatch_reminder_create(
    *,
    cwd: Path,
    scheduler_store: SchedulerStore,
    message: GatewayMessage,
    kind: str,
    schedule_value: str,
    reminder_text: str,
    schedule_label: str,
) -> tuple[dict[str, str], str]:
    if kind == "reminder.once":
        wait_seconds = int(schedule_value)
        schedule = parse_schedule_spec(
            at=(datetime.now(UTC) + timedelta(seconds=wait_seconds)).isoformat(timespec="seconds")
        )
        recurring = False
    else:
        wait_seconds = 60
        schedule = parse_schedule_spec(cron=schedule_value)
        recurring = True
    job = create_scheduler_job(
        store=scheduler_store,
        kind=kind,
        cwd=cwd,
        schedule=schedule,
        payload={
            "text": reminder_text,
            "notify_transport": message.transport,
            "notify_to": message.user_id,
            "notify_chat_id": message.thread_id,
        },
        title=f"reminder-{reminder_text[:40]}",
    )
    scheduler_store.add_job(job)
    _launch_scheduler_watcher(cwd=cwd, wait_seconds=wait_seconds, recurring=recurring)
    return {
        "job_id": job.id,
        "schedule_kind": schedule.kind,
        "schedule_value": schedule.value,
        "next_run_at": job.next_run_at,
        "text": reminder_text,
    }, f"Okay. I'll remind you {schedule_label}: {reminder_text}"


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
            f"- {item['id']} {item['kind']} -> {item['result_status']} ({item['finished_at']})"
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


def _dispatch_workflow_start(
    *,
    cwd: Path,
    goal: str,
) -> tuple[dict[str, str], str]:
    store = WorkflowStore(root=default_workflow_root(cwd))
    title = goal.strip().splitlines()[0][:80] or "Dynamic workflow"
    workflow_id = store.new_id(title)
    run = create_default_workflow(workflow_id=workflow_id, title=title, goal=goal)
    store.add_run(run)
    store.append_event(run.id, kind="workflow.created", message="Workflow created from gateway.")
    return {
        "workflow_id": run.id,
        "status": run.status,
        "node_count": str(len(run.nodes)),
        "workflow_dir": str(store.run_dir(run.id)),
    }, (
        f"Created workflow {run.id} with {len(run.nodes)} defended node(s). "
        f"Run it with: harness workflow resume {run.id} --yes"
    )


def _workflow_store(cwd: Path) -> WorkflowStore:
    return WorkflowStore(root=default_workflow_root(cwd))


def _format_workflow_status(run_id: str, status: str, nodes: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node["status"]] = counts.get(node["status"], 0) + 1
    counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return f"Workflow {run_id}: {status}. Nodes: {counts_text or 'none'}."


def _workflow_status_payload(cwd: Path, workflow_id: str) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    run = store.load_run(workflow_id)
    nodes = [
        {
            "id": node.id,
            "kind": node.kind,
            "role": node.role or node.kind,
            "status": node.status,
            "attempts": str(node.attempts),
        }
        for node in run.nodes
    ]
    payload: dict[str, object] = {
        "workflow_id": run.id,
        "title": run.title,
        "status": run.status,
        "nodes": nodes,
        "updated_at": run.updated_at,
    }
    return payload, _format_workflow_status(run.id, run.status, nodes)


def _workflow_list_payload(cwd: Path) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    runs = store.list_runs()
    items = [
        {
            "workflow_id": run.id,
            "title": run.title,
            "status": run.status,
            "updated_at": run.updated_at,
        }
        for run in runs[:8]
    ]
    if not items:
        return {"workflows": []}, "No workflows found."
    lines = ["Recent workflows:"]
    for item in items:
        lines.append(f"- {item['workflow_id']} {item['status']}: {item['title']}")
    return {"workflows": items}, "\n".join(lines)


def _workflow_report_payload(cwd: Path, workflow_id: str) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    run = store.load_run(workflow_id)
    report_path = store.run_dir(run.id) / "REPORT.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else run.final_report
    text = report or "(no final report yet)"
    if len(text) > 3500:
        suffix = f"\n\nReport truncated. Full report: {report_path}"
        text = text[: 3500 - len(suffix)].rstrip() + suffix
    return {
        "workflow_id": run.id,
        "status": run.status,
        "report_path": str(report_path) if report_path.is_file() else "",
        "report": report,
    }, text


def _workflow_events_payload(
    cwd: Path, workflow_id: str, *, limit: int = 8
) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    events = store.list_events(workflow_id)[-limit:]
    payload = {"workflow_id": workflow_id, "events": [event.to_dict() for event in events]}
    if not events:
        return payload, "No workflow events recorded."
    lines = ["Recent workflow events:"]
    for event in events:
        node = f" {event.node_id}" if event.node_id else ""
        lines.append(f"- {event.kind}{node}: {event.message}")
    return payload, "\n".join(lines)


def _workflow_graph_payload(cwd: Path, workflow_id: str) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    run = store.load_run(workflow_id)
    graph = render_workflow_mermaid(run)
    return {"workflow_id": run.id, "mermaid": graph}, graph


def _workflow_cancel_payload(cwd: Path, workflow_id: str) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    run = store.load_run(workflow_id)
    updated = replace(run, status="cancelled", updated_at=_utcnow_text())
    store.save_run(updated)
    store.append_event(updated.id, kind="workflow.cancelled", message="Workflow cancelled.")
    return {
        "workflow_id": updated.id,
        "status": updated.status,
    }, f"Cancelled workflow {updated.id}."


def _workflow_resume_payload(cwd: Path, workflow_id: str) -> tuple[dict[str, object], str]:
    store = _workflow_store(cwd)
    run = store.load_run(workflow_id)
    log_path = store.run_dir(workflow_id) / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    uv_bin = shutil.which("uv") or "uv"
    command = [
        uv_bin,
        "run",
        "harness",
        "workflow",
        "resume",
        workflow_id,
        "--cwd",
        str(cwd),
        "--yes",
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
    store.append_event(
        workflow_id,
        kind="workflow.launch",
        message=f"Background workflow runner started with pid {process.pid}.",
        data={"pid": process.pid, "log_path": str(log_path)},
    )
    return {
        "workflow_id": run.id,
        "status": run.status,
        "pid": str(process.pid),
        "log_path": str(log_path),
    }, f"Started workflow {run.id} in the background (pid {process.pid})."


def _record_recent_thread(
    profile_threads: list[str], *, thread_id: str, limit: int = 8
) -> list[str]:
    items = [item for item in profile_threads if item != thread_id]
    items.append(thread_id)
    return items[-limit:]


def _upsert_active_work(
    items: list[GatewayWorkRef],
    *,
    work: GatewayWorkRef,
    limit: int = 8,
) -> list[GatewayWorkRef]:
    filtered = [item for item in items if item.ref != work.ref]
    filtered.append(work)
    return filtered[-limit:]


def _save_shared_work(
    *,
    session_store: GatewaySessionStore,
    session: GatewaySessionBinding,
    ref: str,
    kind: str,
    title: str,
    summary: str,
    updated_at: str,
) -> None:
    profile = session_store.get_or_create_profile(
        transport=session.transport,
        user_id=session.user_id,
    )
    active_work = _upsert_active_work(
        profile.active_work,
        work=GatewayWorkRef(
            ref=ref,
            kind=kind,
            title=title,
            summary=summary,
            source_thread_id=session.thread_id,
            updated_at=updated_at,
        ),
    )
    session_store.save_profile(
        replace(
            profile,
            active_work=active_work,
            recent_threads=_record_recent_thread(
                profile.recent_threads, thread_id=session.thread_id
            ),
            updated_at=updated_at,
        )
    )


def _link_session_work(
    session: GatewaySessionBinding,
    *,
    refs: list[str],
    updated_at: str,
) -> GatewaySessionBinding:
    raw_linked = session.metadata.get("linked_work_items", [])
    linked = (
        [str(item).strip() for item in raw_linked if str(item).strip()]
        if isinstance(raw_linked, list)
        else []
    )
    for ref in refs:
        if ref not in linked:
            linked.append(ref)
    return replace(
        session,
        updated_at=updated_at,
        metadata={**session.metadata, "linked_work_items": linked[-8:]},
    )


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
        session = _link_session_work(
            session,
            refs=[f"mission:{mission_id}", f"job:{payload['job_id']}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"mission:{mission_id}",
            kind="mission",
            title=mission_id,
            summary=text,
            updated_at=session.updated_at,
        )
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
        session = _link_session_work(
            session,
            refs=[f"job:{payload['job_id']}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"job:{payload['job_id']}",
            kind="research",
            title=payload["job_id"],
            summary=text,
            updated_at=session.updated_at,
        )
        reply = GatewayReply(
            session_id=session.id,
            command="research.start",
            status="ok",
            text=text,
            data=payload,
        )
    elif command == "workflow" and len(tokens) >= 3 and lowered[1] == "start":
        goal = message.text.split(None, 2)[2].strip()
        payload, text = _dispatch_workflow_start(cwd=cwd, goal=goal)
        session = replace(
            session,
            last_run_id=payload["workflow_id"],
            last_command="workflow start",
            updated_at=_utcnow_text(),
        )
        session = _link_session_work(
            session,
            refs=[f"workflow:{payload['workflow_id']}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"workflow:{payload['workflow_id']}",
            kind="workflow",
            title=goal[:80],
            summary=text,
            updated_at=session.updated_at,
        )
        reply = GatewayReply(
            session_id=session.id,
            command="workflow.start",
            status="ok",
            text=text,
            data=payload,
        )
    elif command == "workflow" and len(tokens) >= 2 and lowered[1] == "list":
        payload, text = _workflow_list_payload(cwd)
        session = replace(
            session,
            last_command="workflow list",
            updated_at=_utcnow_text(),
        )
        session_store.save_session(session)
        reply = GatewayReply(
            session_id=session.id,
            command="workflow.list",
            status="ok",
            text=text,
            data=payload,
        )
    elif (
        command == "workflow"
        and len(tokens) >= 3
        and lowered[1]
        in {
            "status",
            "resume",
            "report",
            "cancel",
            "events",
            "graph",
        }
    ):
        workflow_id = tokens[2]
        if lowered[1] == "status":
            payload, text = _workflow_status_payload(cwd, workflow_id)
        elif lowered[1] == "resume":
            payload, text = _workflow_resume_payload(cwd, workflow_id)
        elif lowered[1] == "report":
            payload, text = _workflow_report_payload(cwd, workflow_id)
        elif lowered[1] == "cancel":
            payload, text = _workflow_cancel_payload(cwd, workflow_id)
        elif lowered[1] == "events":
            payload, text = _workflow_events_payload(cwd, workflow_id)
        else:
            payload, text = _workflow_graph_payload(cwd, workflow_id)
        session = replace(
            session,
            last_run_id=workflow_id,
            last_command=f"workflow {lowered[1]}",
            updated_at=_utcnow_text(),
        )
        session = _link_session_work(
            session,
            refs=[f"workflow:{workflow_id}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"workflow:{workflow_id}",
            kind=f"workflow-{lowered[1]}",
            title=workflow_id,
            summary=text[:500],
            updated_at=session.updated_at,
        )
        reply = GatewayReply(
            session_id=session.id,
            command=f"workflow.{lowered[1]}",
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
        session = _link_session_work(
            session,
            refs=[f"mission:{mission_id}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        text = report.summary
        if report.next_actions:
            text += "\nNext: " + "; ".join(report.next_actions)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"mission:{mission_id}",
            kind="mission-report",
            title=mission_id,
            summary=report.summary,
            updated_at=session.updated_at,
        )
        reply = GatewayReply(
            session_id=session.id,
            command="report",
            status="ok",
            text=text,
            data=report.to_dict(),
        )
    elif reminder := _parse_reminder_intent(message.text):
        kind, schedule_value, reminder_text, schedule_label = reminder
        payload, text = _dispatch_reminder_create(
            cwd=cwd,
            scheduler_store=scheduler_store,
            message=message,
            kind=kind,
            schedule_value=schedule_value,
            reminder_text=reminder_text,
            schedule_label=schedule_label,
        )
        session = replace(
            session,
            last_job_id=payload["job_id"],
            last_command="reminder.create",
            updated_at=_utcnow_text(),
        )
        session = _link_session_work(
            session,
            refs=[f"job:{payload['job_id']}"],
            updated_at=session.updated_at,
        )
        session_store.save_session(session)
        _save_shared_work(
            session_store=session_store,
            session=session,
            ref=f"job:{payload['job_id']}",
            kind="reminder",
            title=reminder_text,
            summary=text,
            updated_at=session.updated_at,
        )
        reply = GatewayReply(
            session_id=session.id,
            command="reminder.create",
            status="ok",
            text=text,
            data=payload,
        )
    else:
        reply = GatewayReply(
            session_id=session.id,
            command="unknown",
            status="error",
            text=(
                "Unsupported gateway command. Use one of: status, runs, "
                "mission start <id>, research start, workflow start <goal>, "
                "workflow status|resume|report|cancel|events|graph <id>, workflow list, "
                "approve <id>, report <mission_id>."
            ),
        )
    if reply.command not in {
        "mission.start",
        "research.start",
        "workflow.start",
        "approve",
        "report",
        "reminder.create",
    }:
        session = replace(session, last_command=reply.command, updated_at=_utcnow_text())
        session_store.save_session(session)
    for hook in hooks:
        hook.on_gateway_reply(cwd=cwd, message=message, reply=reply)
    return reply, session


__all__ = ["dispatch_gateway_message", "is_gateway_control_message"]
