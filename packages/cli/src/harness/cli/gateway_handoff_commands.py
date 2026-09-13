"""Transcript handoff; caller holds the destination/source conversation lock."""

from __future__ import annotations

import json
from dataclasses import replace

from harness.core import Session
from harness.core.gateway_handoffs import HandoffStore
from harness.core.gateway_models import GatewayRuntimeBinding
from harness.core.trajectories import validate_messages


async def handoff_command(
    *, command, argument, cwd, owner, conversation, session, binding, sessions, storage, reply
):
    if command == "/whoami":
        return reply(json.dumps(owner), data=owner)
    store = HandoffStore(sessions.root)
    try:
        if command == "/handoff":
            if argument.startswith("revoke "):
                revoked = store.revoke(argument.removeprefix("revoke ").strip(), owner)
                return reply(
                    "Handoff revoked." if revoked else "No pending handoff with that code.",
                    status="ok" if revoked else "unavailable",
                )
            target = json.loads(argument)
            HandoffStore.identity(target)
            if session is None or binding is None:
                return reply("Start a conversation before handing it off.", status="empty")
            expected_scope = json.dumps(
                [owner["transport"], owner["user_id"]], separators=(",", ":"), ensure_ascii=False
            )
            scope = session.metadata.get("memory_scope", {})
            if scope.get("user_id") != expected_scope:
                return reply(
                    "Only the scoped conversation of this remote identity can be handed off.",
                    status="forbidden",
                )
            validate_messages(session.messages, require_complete=True)
            # Configuration/persona/system instructions, memories and approval
            # ledgers stay with their owner. Completed tool evidence is context.
            messages = []
            for message in session.messages:
                if message.role == "system":
                    if (message.content or "").startswith("[Compacted context summary]"):
                        messages.append(
                            {
                                "role": "user",
                                "content": "Earlier conversation summary:\n" + message.content,
                            }
                        )
                    continue
                messages.append(message.model_dump(mode="json"))
            token = store.issue(
                owner,
                target,
                {
                    "provider": binding.provider,
                    "model": binding.model,
                    "messages": messages,
                    "source_session_id": session.id,
                },
            )
            return reply(
                "In the exact destination shown, send:\n/continue "
                + token
                + "\nThe code expires in 10 minutes. This copies the conversation and its attachments; memories, approvals and host permissions stay with their owner.",
                data={"code": token, "destination": target, "expires_in": 600},
            )
        if command == "/continue":
            claim = store.claim(argument, owner)
            new_id = claim["session_id"]
            if claim["complete"]:
                return reply(
                    "This handoff was already imported.", data={"harness_session_id": new_id}
                )
            payload = claim["payload"]
            copied = await storage.get(new_id)
            if copied is None:
                copied = Session(
                    id=new_id,
                    cwd=cwd,
                    provider=payload["provider"],
                    model=payload["model"],
                    status="done",
                    messages=payload["messages"],
                    metadata={
                        "memory_scope": {
                            "workspace": str(cwd.resolve()),
                            "user_id": json.dumps(
                                [owner["transport"], owner["user_id"]],
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                        "handoff_source_session_id": payload["source_session_id"],
                    },
                )
                await storage.save(copied)
            sessions.bind_runtime_session(
                GatewayRuntimeBinding(
                    session_id=new_id,
                    gateway_session_id=conversation.id,
                    **owner,
                    provider=copied.provider,
                    model=copied.model,
                )
            )
            from harness.cli.gateway_runtime import _runtime_session_key

            runtime_key = _runtime_session_key(provider=copied.provider, model=copied.model)
            metadata = {
                key: value
                for key, value in conversation.metadata.items()
                if not key.startswith("harness_session_id")
            }
            metadata.update(
                harness_session_id=new_id,
                harness_session_base_id=new_id,
                provider_override=copied.provider,
                model_override=copied.model,
                thread_context=[],
                thread_summary="",
            )
            metadata["harness_session_id_" + runtime_key] = new_id
            sessions.save_session(replace(conversation, metadata=metadata))
            store.complete(argument, owner)
            return reply(
                "Conversation imported. Your next message will continue it with the permissions of this destination.",
                data={"harness_session_id": new_id},
            )
        return reply("Unsupported handoff command.", status="invalid")
    except (ValueError, TypeError):
        return reply(
            "Invalid or unavailable handoff. Use /whoami at the destination, then /handoff with that exact JSON in the source conversation, followed by /continue CODE at the destination.",
            status="invalid",
        )
    finally:
        store.close()
