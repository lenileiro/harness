"""Scheduled prompt execution through the same scoped gateway approval boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import typer

from harness.cli import gateway_runtime
from harness.cli.config import load_config
from harness.core.gateway_evidence import approval_request_text
from harness.core.gateway_models import GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_router import gateway_pending_approvals
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler.models import SchedulerExecutionResult, SchedulerJob
from harness.storage.sqlite import SQLiteStorage


def prompt_session_id(job_id: str) -> str:
    return f"sess_prompt_{hashlib.sha256(job_id.encode()).hexdigest()[:24]}"


def _result(
    job: SchedulerJob,
    *,
    status: str,
    reason: str,
    text: str,
    approval_ids: list[str],
    notify: bool = True,
    retry_after_seconds: float | None = None,
) -> SchedulerExecutionResult:
    directory = Path(job.cwd) / ".harness" / "scheduler" / "prompt-results" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job.id,
        "session_id": prompt_session_id(job.id),
        "status": status,
        "stop_reason": reason,
        "text": text,
        "approval_ids": approval_ids,
        "provider": str(job.payload.get("provider") or ""),
        "model": str(job.payload.get("model") or ""),
        "transport": str(job.payload.get("transport") or "local"),
        "user_id": str(job.payload.get("user_id") or "cli"),
        "thread_id": str(job.payload.get("thread_id") or job.id),
        "local_only": bool(job.payload.get("local_only", False)),
    }
    if payload["local_only"]:
        payload["resume_commands"] = [
            *[f"harness approvals grant {approval_id}" for approval_id in approval_ids],
            f"harness sessions resume {payload['session_id']}",
        ]
    (directory / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SchedulerExecutionResult(
        status=status,
        stop_reason=reason,
        record_dir=str(directory),
        notification_text=text,
        notify_result=notify,
        retry_after_seconds=retry_after_seconds,
    )


async def run_scheduled_prompt(job: SchedulerJob) -> SchedulerExecutionResult:
    from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT

    cwd = Path(job.cwd)
    payload = job.payload
    prompt = str(payload.get("prompt") or "").strip()
    provider = str(payload.get("provider") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not prompt or not provider or not model:
        raise ValueError("Scheduled prompts require a prompt, provider and model")
    if provider == "codex":
        raise ValueError(
            "Codex native tools cannot use the remote approval boundary; use an API provider"
        )
    transport = str(payload.get("transport") or "local")
    user_id = str(payload.get("user_id") or "cli")
    thread_id = str(payload.get("thread_id") or job.id)
    session_id = prompt_session_id(job.id)
    sessions = GatewaySessionStore(root=default_gateway_root(cwd))
    with sessions.conversation_lock(
        transport=transport, user_id=user_id, thread_id=thread_id
    ) as acquired:
        if not acquired:
            return _result(
                job,
                status="blocked",
                reason="conversation_busy",
                text="Scheduled prompt deferred: its conversation is busy.",
                approval_ids=[],
                notify=False,
                retry_after_seconds=30,
            )
        owner = sessions.get_or_create_session(
            transport=transport, user_id=user_id, thread_id=thread_id
        )
        binding = GatewayRuntimeBinding(
            session_id=session_id,
            gateway_session_id=owner.id,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
            provider=provider,
            model=model,
            max_steps=int(payload.get("max_steps", 20)),
            local_only=bool(payload.get("local_only", False)),
        )
        sessions.bind_runtime_session(binding)
        storage = SQLiteStorage(path=cwd / ".harness" / "harness.db")
        try:
            pending = [
                item
                for item in await gateway_pending_approvals(
                    approval_store=storage,
                    session_store=sessions,
                    transport=transport,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                if item.session_id == session_id
            ]
            if pending:
                return _result(
                    job,
                    status="approval_required",
                    reason="waiting_for_approval",
                    text="\n\n".join(approval_request_text(item) for item in pending),
                    approval_ids=[item.id for item in pending],
                    notify=job.last_status != "approval_required",
                )
            previous = await storage.get(session_id)
            granted = await storage.list_unreplayed_granted(session_id=session_id)
            if granted:
                prompt = "Continue the existing scheduled task from its approved tool result. Do not repeat a completed action."
            elif (
                previous is not None
                and previous.status == "paused"
                and job.last_status == "approval_required"
            ):
                return _result(
                    job,
                    status="denied",
                    reason="approval_denied_or_expired",
                    text="The scheduled action was denied or its approval expired. It was not executed.",
                    approval_ids=[],
                )
            config_path = str(payload.get("config_path") or "")
            cfg = load_config(Path(config_path) if config_path else None)
            try:
                text = await asyncio.wait_for(
                    gateway_runtime._run_gateway_chat_turn(
                        cwd=cwd,
                        prompt=prompt,
                        chain=[binding.provider],
                        model=binding.model,
                        session_id=session_id,
                        max_steps=binding.max_steps,
                        config=cfg,
                        system_prompt=_DEFAULT_SYSTEM_PROMPT,
                        transport=transport,
                        user_id=user_id,
                        local_only=binding.local_only,
                    ),
                    timeout=float(payload.get("timeout_seconds", 300)),
                )
                status, reason = "completed", "prompt_completed"
            except (typer.Exit, TimeoutError) as exc:
                text = (
                    await gateway_runtime._latest_run_failure_reply(cwd=cwd, session_id=session_id)
                    or f"Scheduled prompt stopped: {type(exc).__name__}. Inspect its session before requesting another action."
                )
                status, reason = "failed", "prompt_stopped"
            pending = [
                item
                for item in await gateway_pending_approvals(
                    approval_store=storage,
                    session_store=sessions,
                    transport=transport,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                if item.session_id == session_id
            ]
            if pending:
                text = "\n\n".join(approval_request_text(item) for item in pending)
                status, reason = "approval_required", "waiting_for_approval"
            return _result(
                job,
                status=status,
                reason=reason,
                text=text,
                approval_ids=[item.id for item in pending],
            )
        finally:
            await storage.close()
