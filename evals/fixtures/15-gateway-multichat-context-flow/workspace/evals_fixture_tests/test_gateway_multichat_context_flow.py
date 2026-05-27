from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from harness.core.gateway_models import GatewayMessage
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler_store import SchedulerStore


class _PopenStub:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def _require_symbol(module_name: str, symbol: str):
    module = importlib.import_module(module_name)
    value = getattr(module, symbol, None)
    if value is None:
        pytest.fail(f"{symbol} missing from {module_name}")
    return value


def test_shared_user_profile_persists_work_across_threads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("harness.core.gateway_router.subprocess.Popen", _PopenStub)
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    scheduler_store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    if not hasattr(session_store, "save_profile"):
        pytest.fail("GatewaySessionStore.save_profile is missing")
    if not hasattr(session_store, "load_profile"):
        pytest.fail("GatewaySessionStore.load_profile is missing")
    if not hasattr(session_store, "list_user_sessions"):
        pytest.fail("GatewaySessionStore.list_user_sessions is missing")

    async def _run() -> None:
        reply, session = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id="msg-1",
                transport="whatsapp",
                user_id="15551234567",
                thread_id="thread-a",
                text="remind me in 5 minutes to check the build",
            ),
        )
        assert reply.command == "reminder.create"
        assert session.last_job_id
        assert session.metadata["linked_work_items"]

    asyncio.run(_run())

    profile = session_store.load_profile("whatsapp", "15551234567")
    assert profile.transport == "whatsapp"
    assert profile.user_id == "15551234567"
    assert profile.active_work
    work = profile.active_work[0]
    assert work.kind == "reminder"
    assert "check the build" in work.summary
    assert work.source_thread_id == "thread-a"
    assert "thread-a" in profile.recent_threads
    assert session_store.list_user_sessions(transport="whatsapp", user_id="15551234567")


def test_gateway_converse_includes_shared_work_and_other_thread_context(
    tmp_path: Path, monkeypatch
) -> None:
    session_store = GatewaySessionStore(root=tmp_path / ".harness" / "gateway")
    captured: dict[str, object] = {}
    GatewayUserProfile = _require_symbol("harness.core.gateway_models", "GatewayUserProfile")
    GatewayWorkRef = _require_symbol("harness.core.gateway_models", "GatewayWorkRef")

    profile = GatewayUserProfile(
        id=session_store.get_or_create_profile(transport="whatsapp", user_id="15551234567").id,
        transport="whatsapp",
        user_id="15551234567",
        active_work=[
            GatewayWorkRef(
                ref="job-1",
                kind="reminder",
                title="check the build",
                summary="Reminder scheduled to check the build",
                source_thread_id="thread-a",
            )
        ],
        recent_threads=["thread-a"],
    )
    session_store.save_profile(profile)
    thread_a = session_store.get_or_create_session(
        transport="whatsapp",
        user_id="15551234567",
        thread_id="thread-a",
    )
    thread_a.metadata["thread_summary"] = "Reminder thread about checking the build."
    session_store.save_session(thread_a)

    async def _fake_run_once(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Continuing the build reminder context."

    monkeypatch.setattr("harness.cli.gateway_commands._run_once_impl", _fake_run_once)

    from harness.cli.gateway_commands import _run_gateway_conversation

    payload = asyncio.run(
        _run_gateway_conversation(
            cwd=tmp_path,
            session_store=session_store,
            transport="whatsapp",
            user_id="15551234567",
            thread_id="thread-b",
            message="continue that",
            max_steps=1,
        )
    )

    prompt = str(captured["prompt"])
    assert "continue that" in prompt
    assert "check the build" in prompt
    assert "Reminder thread about checking the build." in prompt
    assert payload["reply"]["text"] == "Continuing the build reminder context."
