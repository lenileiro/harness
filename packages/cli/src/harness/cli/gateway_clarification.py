"""Scoped gateway text answers for durable clarification questions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from harness.cli.common import _load_cli_config
from harness.core import default_gateway_root
from harness.core.clarification import (
    PendingQuestion,
    QuestionStore,
    parse_question_answer,
    render_question,
)
from harness.core.gateway_models import GatewayRuntimeBinding, GatewaySessionBinding
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.memory import MemoryScope


def gateway_question_scope(
    cwd: Path, transport: str, user_id: str, *, local_only: bool = False
) -> MemoryScope:
    return MemoryScope(
        workspace=str(cwd),
        user_id=None
        if local_only
        else json.dumps([transport, user_id], separators=(",", ":"), ensure_ascii=False),
    )


async def gateway_questions(
    *, cwd: Path, session_store: GatewaySessionStore, transport: str, user_id: str, thread_id: str
) -> list[tuple[PendingQuestion, GatewayRuntimeBinding]]:
    bindings = session_store.list_runtime_bindings(
        transport=transport, user_id=user_id, thread_id=thread_id
    )
    if not bindings:
        return []
    store = await asyncio.to_thread(QuestionStore, cwd / ".harness" / "harness.db")
    try:
        return [
            (question, binding)
            for binding in bindings
            for question in await asyncio.to_thread(
                store.list_pending,
                scope=gateway_question_scope(
                    cwd, transport, user_id, local_only=binding.local_only
                ),
                session_id=binding.session_id,
            )
        ]
    finally:
        await asyncio.to_thread(store.close)


def question_data(record: PendingQuestion) -> dict[str, Any]:
    return {
        "question_id": record.id,
        "harness_session_id": record.session_id,
        "questions": [
            {"id": f"q{index}", **question.model_dump(mode="json")}
            for index, question in enumerate(record.questions)
        ],
        "answers": record.answers,
        "expires_at": record.expires_at.isoformat(),
        "question_status": record.status,
    }


def question_reply(record: PendingQuestion, session: GatewaySessionBinding) -> dict[str, object]:
    return {
        "session": session.to_dict(),
        "reply": {
            "session_id": session.id,
            "command": "clarify",
            "status": "question_required",
            "text": render_question(record),
            "data": question_data(record),
        },
    }


async def resume_gateway_question(
    *,
    cwd: Path,
    session_store: GatewaySessionStore,
    record: PendingQuestion,
    binding: GatewayRuntimeBinding,
) -> dict[str, object]:
    from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT
    from harness.cli.gateway_runtime import (
        _gateway_pending_reply,
        _gateway_turn_timeout_seconds,
        _latest_gateway_attachments,
        _run_gateway_chat_turn,
    )

    session = session_store.load_session(binding.gateway_session_id)
    approval_text, approval_ids = await _gateway_pending_reply(
        cwd=cwd,
        session_store=session_store,
        transport=binding.transport,
        user_id=binding.user_id,
        thread_id=binding.thread_id,
    )
    if approval_ids:
        return {
            "session": session.to_dict(),
            "reply": {
                "session_id": session.id,
                "command": "answer",
                "status": "approval_required",
                "text": approval_text,
                "data": {
                    "harness_session_id": record.session_id,
                    "question_id": record.id,
                    "approval_ids": approval_ids,
                },
            },
        }
    text = await asyncio.wait_for(
        _run_gateway_chat_turn(
            cwd=cwd,
            prompt="Continue the original task using the recorded clarification result. A question answer does not grant tool approval.",
            chain=[binding.provider],
            model=binding.model,
            session_id=binding.session_id,
            max_steps=binding.max_steps,
            config=_load_cli_config(None),
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            transport=binding.transport,
            user_id=binding.user_id,
            local_only=binding.local_only,
        ),
        timeout=_gateway_turn_timeout_seconds(),
    )
    pending = await gateway_questions(
        cwd=cwd,
        session_store=session_store,
        transport=binding.transport,
        user_id=binding.user_id,
        thread_id=binding.thread_id,
    )
    if pending:
        return question_reply(pending[0][0], session)
    approval_text, approval_ids = await _gateway_pending_reply(
        cwd=cwd,
        session_store=session_store,
        transport=binding.transport,
        user_id=binding.user_id,
        thread_id=binding.thread_id,
    )
    return {
        "session": session.to_dict(),
        "reply": {
            "session_id": session.id,
            "command": "answer",
            "status": "approval_required" if approval_ids else "ok",
            "text": approval_text
            or text
            or "The continuation stopped; inspect the original session.",
            "data": {
                "harness_session_id": record.session_id,
                "question_id": record.id,
                "approval_ids": approval_ids,
                "attachments": await _latest_gateway_attachments(
                    cwd=cwd, session_id=record.session_id
                ),
            },
        },
    }


async def dispatch_gateway_question(
    *, cwd: Path, message: str, transport: str, user_id: str, thread_id: str
) -> dict[str, object] | None:
    parts = message.strip().split(None, 2)
    command = parts[0].lower().lstrip("/") if parts else ""
    if command not in {"questions", "answer", "skip-question"}:
        return None
    sessions = GatewaySessionStore(root=default_gateway_root(cwd))
    session = sessions.get_or_create_session(
        transport=transport, user_id=user_id, thread_id=thread_id
    )

    def response(text: str, status: str = "ok") -> dict[str, object]:
        return {
            "session": session.to_dict(),
            "reply": {
                "session_id": session.id,
                "command": command,
                "status": status,
                "text": text,
                "data": {},
            },
        }

    with sessions.conversation_lock(
        transport=transport, user_id=user_id, thread_id=thread_id
    ) as acquired:
        if not acquired:
            return response(
                "This conversation is busy. Retry when its current turn finishes.", "busy"
            )
        pending = await gateway_questions(
            cwd=cwd,
            session_store=sessions,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
        )
        if command == "questions":
            return (
                question_reply(pending[0][0], session)
                if pending
                else response("No pending clarification questions in this conversation.")
            )
        selected = next(
            (
                (record, binding)
                for record, binding in pending
                if len(parts) > 1 and record.id == parts[1]
            ),
            None,
        )
        if selected is None:
            return response(
                "Question not found in this conversation. Use /questions to inspect pending questions.",
                "error",
            )
        record, binding = selected
        store = await asyncio.to_thread(QuestionStore, cwd / ".harness" / "harness.db")
        try:
            if command == "skip-question":
                record = await asyncio.to_thread(
                    store.cancel, record.id, scope=record.scope, session_id=binding.session_id
                )
            elif record.status == "pending":
                if len(parts) < 3:
                    return response(
                        'Usage: /answer QUESTION_ID TEXT or /answer QUESTION_ID {"q0":"answer"}',
                        "error",
                    )
                record = await asyncio.to_thread(
                    store.answer,
                    record.id,
                    scope=record.scope,
                    session_id=binding.session_id,
                    answers=parse_question_answer(record, parts[2]),
                )
        except ValueError as exc:
            return response(str(exc), "error")
        finally:
            await asyncio.to_thread(store.close)
        if record.status == "pending":
            return question_reply(record, session)
        return await resume_gateway_question(
            cwd=cwd, session_store=sessions, record=record, binding=binding
        )
