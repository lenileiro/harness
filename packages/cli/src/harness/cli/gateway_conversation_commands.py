"""Explicit, owner-scoped conversation commands shared by messaging transports."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from inspect import isawaitable
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness.cli.common import _build_adapter, _load_cli_config
from harness.cli.runtime_helpers import build_storage
from harness.core.activity import ActivityStore
from harness.core.approval import ApprovalStore
from harness.core.budget import count_tokens
from harness.core.compactor import ContextCompactor
from harness.core.gateway_models import default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore

CONVERSATION_COMMANDS = frozenset(
    {"/model", "/retry", "/undo", "/compress", "/usage", "/whoami", "/handoff", "/continue"}
)


def is_conversation_command(text: str) -> bool:
    words = text.strip().split(maxsplit=1)
    return bool(words and words[0].lower() in CONVERSATION_COMMANDS)


class _CompactionAdapter:
    """Give stateful adapters an isolated lifecycle with no executable tools."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.name = adapter.name
        self.session_id = "compact_" + uuid4().hex

    def stream(self, *, model, messages, tools=None, **kwargs):
        return self.adapter.stream(
            model=model, messages=messages, tools=[], session_id=self.session_id, **kwargs
        )

    async def capabilities(self):
        return await self.adapter.capabilities()

    async def cancel(self, session_id: str) -> None:
        await self.adapter.cancel(session_id)

    async def close(self) -> None:
        end = getattr(self.adapter, "end_run", None)
        if callable(end):
            result = end(self.session_id)
            if isawaitable(result):
                await result


async def conversation_command(
    *, cwd: Path, message: str, transport: str, user_id: str, thread_id: str
) -> dict[str, object]:
    # Imported lazily to keep the gateway entrypoint/command dispatcher acyclic.
    from harness.cli import gateway_runtime as runtime

    command, *rest = message.strip().split(maxsplit=1)
    command = command.lower()
    argument = rest[0].strip() if rest else ""
    sessions = GatewaySessionStore(root=default_gateway_root(cwd))
    owner = {"transport": transport, "user_id": user_id, "thread_id": thread_id}
    with sessions.conversation_lock(**owner) as acquired:
        conversation = sessions.get_or_create_session(**owner)

        def reply(
            text: str, *, status: str = "ok", data: dict[str, Any] | None = None
        ) -> dict[str, object]:
            return {
                "reply": {
                    "session_id": conversation.id,
                    "command": command[1:],
                    "status": status,
                    "text": text,
                    "data": data or {},
                },
                "session": conversation.to_dict(),
            }

        if not acquired:
            return reply(
                "This conversation is busy. Wait for its active turn to finish.", status="busy"
            )
        if command not in CONVERSATION_COMMANDS or (
            argument and command not in {"/model", "/handoff", "/continue"}
        ):
            return reply(
                "Use /model [MODEL], /retry, /undo, /compress, /usage, /whoami, /handoff DESTINATION_JSON or /continue CODE.",
                status="invalid",
            )
        if command == "/whoami":
            return reply(json.dumps(owner), data=owner)
        storage = build_storage(db=cwd / ".harness/harness.db", in_memory=False, cwd=cwd)
        try:
            identifier = str(conversation.metadata.get("harness_session_id") or "")
            binding = sessions.load_runtime_binding(identifier) if identifier else None
            if binding is not None and not binding.belongs_to(**owner):
                return reply(
                    "The active session is not owned by this conversation.", status="forbidden"
                )
            session = await storage.get(identifier) if binding is not None else None
            if session is not None and await asyncio.to_thread(
                session.cwd.resolve
            ) != await asyncio.to_thread(cwd.resolve):
                return reply(
                    "The active session belongs to a different workspace.", status="forbidden"
                )
            if command == "/model" and not argument:
                config = _load_cli_config(None)
                model = (
                    session.model
                    if session
                    else conversation.metadata.get("model_override")
                    or config.default_model
                    or "default"
                )
                provider = binding.provider if binding else config.default_provider or "default"
                return reply(
                    f"Model: {provider}/{model}", data={"provider": provider, "model": model}
                )
            if session is None and command not in {"/model", "/continue"}:
                return reply("Start a conversation before using this command.", status="empty")
            if session is not None and binding is not None:
                if command == "/usage":
                    if not isinstance(storage, ActivityStore):
                        return reply(
                            "This storage does not provide recorded usage.", status="unavailable"
                        )
                    events = await storage.list_activity(
                        session_id=session.id, kinds=("usage.recorded",), limit=100000
                    )
                    totals = {
                        key: sum(int(event.data.get(key, 0)) for event in events)
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    }
                    return reply(
                        json.dumps(totals),
                        data={
                            "session_id": session.id,
                            "usage": totals,
                            "records": len(events),
                            "truncated": len(events) == 100000,
                        },
                    )
                if session.metadata.get("pending_question_id"):
                    return reply(
                        "Answer or skip the pending question and continue the original session before changing this conversation.",
                        status="question_required",
                    )
                if session.status in {"running", "paused"}:
                    return reply(
                        "Finish the active turn or resolve its approvals before changing this conversation.",
                        status="busy",
                    )
                if isinstance(storage, ApprovalStore):
                    pending = await storage.list_approvals(
                        session_id=session.id, status="pending", limit=1
                    )
                    granted = await storage.list_unreplayed_granted(session_id=session.id)
                    if pending or granted:
                        return reply(
                            "Resolve pending or unfinished approvals before changing this conversation.",
                            status="approval_required",
                        )
            if command in {"/handoff", "/continue"}:
                from harness.cli.gateway_handoff_commands import handoff_command

                return await handoff_command(
                    command=command,
                    argument=argument,
                    cwd=cwd,
                    owner=owner,
                    conversation=conversation,
                    session=session,
                    binding=binding,
                    sessions=sessions,
                    storage=storage,
                    reply=reply,
                )
            if command == "/model":
                if len(argument) > 256 or any(character.isspace() for character in argument):
                    return reply(
                        "Provide one model identifier of at most 256 characters.", status="invalid"
                    )
                metadata = {**conversation.metadata, "model_override": argument}
                if session is not None and binding is not None:
                    new_id = "sess_" + uuid4().hex
                    copied = session.model_copy(
                        deep=True,
                        update={
                            "id": new_id,
                            "model": argument,
                            "forked_from": session.id,
                            "status": "done",
                            "created_at": datetime.now(UTC),
                            "updated_at": datetime.now(UTC),
                        },
                    )
                    new_binding = replace(binding, session_id=new_id, model=argument)
                    sessions.bind_runtime_session(new_binding)
                    await storage.save(copied)
                    runtime_key = runtime._runtime_session_key(
                        provider=binding.provider, model=argument
                    )
                    metadata.update(harness_session_id=new_id)
                    metadata[f"harness_session_id_{runtime_key}"] = new_id
                conversation = replace(conversation, metadata=metadata)
                sessions.save_session(conversation)
                return reply(
                    f"Model selected: {argument}. The next turn will use it.",
                    data={"model": argument},
                )
            assert session is not None and binding is not None
            if command == "/undo":
                start = next(
                    (
                        index
                        for index in range(len(session.messages) - 1, -1, -1)
                        if session.messages[index].role == "user"
                    ),
                    None,
                )
                if start is None:
                    return reply("There is no user turn to undo.", status="empty")
                session.metadata["undo_archive"] = [
                    item.model_dump(mode="json") for item in session.messages[start:]
                ]
                session.messages = session.messages[:start]
                session.touch()
                await storage.save(session)
                conversation = replace(
                    conversation,
                    metadata={**conversation.metadata, "thread_context": [], "thread_summary": ""},
                )
                sessions.save_session(conversation)
                return reply("Removed the last conversation turn. Workspace changes remain.")
            if command == "/retry":
                previous = next(
                    (item for item in reversed(session.messages) if item.role == "user"), None
                )
                if previous is None:
                    return reply("There is no user prompt to retry.", status="empty")
                from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT

                result = await asyncio.wait_for(
                    runtime._run_gateway_chat_turn(
                        cwd=cwd,
                        prompt=previous.content or "",
                        attachments=previous.attachments,
                        chain=[binding.provider],
                        model=binding.model,
                        session_id=session.id,
                        max_steps=binding.max_steps,
                        config=_load_cli_config(None),
                        system_prompt=_DEFAULT_SYSTEM_PROMPT,
                        transport=transport,
                        user_id=user_id,
                    ),
                    timeout=runtime._gateway_turn_timeout_seconds(),
                )
                pending_text, pending_ids = await runtime._gateway_pending_reply(
                    cwd=cwd, session_store=sessions, **owner
                )
                return reply(
                    pending_text or result,
                    status="approval_required" if pending_ids else "ok",
                    data={
                        "harness_session_id": session.id,
                        "approval_ids": pending_ids,
                        "attachments": await runtime._latest_gateway_attachments(
                            cwd=cwd, session_id=session.id
                        ),
                    },
                )
            if command == "/compress":
                config = _load_cli_config(None)
                if binding.provider == "codex":
                    settings = config.provider("codex")
                    if settings.get("mode") == "exec":
                        return reply(
                            "Remote compression requires Codex app-server mode; explicit exec mode is unavailable here.",
                            status="unavailable",
                        )
                    config = replace(
                        config,
                        provider_settings={
                            **config.provider_settings,
                            "codex": {**settings, "mode": "app-server"},
                        },
                    )
                adapter = _CompactionAdapter(
                    _build_adapter(binding.provider, base_url=None, config=config)
                )
                try:
                    compactor = ContextCompactor(
                        adapter=adapter, model=session.model, keep_recent_tokens=2000
                    )
                    compacted = await asyncio.wait_for(
                        compactor.compact(session.messages),
                        timeout=runtime._gateway_turn_timeout_seconds(),
                    )
                finally:
                    await adapter.close()
                if compacted is session.messages or count_tokens(
                    compacted, session.model
                ) >= count_tokens(session.messages, session.model):
                    return reply(
                        "Conversation unchanged: too little old context, or summarization was unavailable.",
                        status="unchanged",
                    )
                session.metadata["pre_compaction_archive"] = [
                    item.model_dump(mode="json") for item in session.messages
                ]
                session.messages = compacted
                session.touch()
                await storage.save(session)
                return reply(
                    f"Conversation compressed to {len(compacted)} messages; the original transcript is archived."
                )
            return reply("Unsupported conversation command.", status="invalid")
        except Exception:
            return reply(
                "The conversation command could not finish. Inspect the current session before retrying; completed actions are not rolled back.",
                status="uncertain" if command == "/retry" else "error",
            )
        finally:
            close = getattr(storage, "close", None)
            if callable(close):
                result = close()
                if isawaitable(result):
                    await result
