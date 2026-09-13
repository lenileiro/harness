from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from harness.core.gateway_models import GatewayMessage, GatewayUserProfile
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler_models import SchedulerRunRecord
from harness.core.scheduler_runtime import create_scheduler_job, parse_schedule_spec
from harness.core.scheduler_store import SchedulerStore


@pytest.mark.parametrize(
    ("kind", "local_only"),
    [
        ("prompt.run", False),
        ("prompt.run", True),
        ("reminder.once", False),
        ("reminder.recurring", False),
    ],
)
def test_gateway_runs_filters_private_legacy_ids_by_exact_owner(
    tmp_path: Path, kind: str, local_only: bool
) -> None:
    sessions = GatewaySessionStore(root=tmp_path / ".harness/gateway")
    store = SchedulerStore(root=tmp_path / ".harness/scheduler")
    payload: dict[str, object] = (
        {
            "transport": "test",
            "user_id": "alice",
            "thread_id": "private",
            "prompt": "secret",
            "local_only": local_only,
        }
        if kind == "prompt.run"
        else {
            "notify_transport": "test",
            "notify_to": "alice",
            "notify_chat_id": "private",
            "text": "secret",
        }
    )
    job = create_scheduler_job(
        store=store,
        kind=kind,
        cwd=tmp_path,
        schedule=parse_schedule_spec(every="1h"),
        title="secret-acquisition-plan",
        payload=payload,
    )
    store.add_job(job)
    record = SchedulerRunRecord(
        id="run-private",
        job_id=job.id,
        kind=kind,
        cwd=str(tmp_path),
        trigger="test",
        status="completed",
        result_status="completed",
        result_stop_reason="secret response",
        started_at="2026-09-13T10:00:00+00:00",
        finished_at="2026-09-13T10:01:00+00:00",
        record_dir="secret-path",
        summary="secret response",
    )
    store.add_run_record(record)
    store.add_run_record(
        replace(record, id="run-public", job_id="job-mission", kind="mission.schedule_once")
    )

    async def check() -> None:
        for transport, user_id, thread_id in [
            ("test", "bob", "private"),
            ("other", "alice", "private"),
            ("test", "alice", "other-thread"),
        ]:
            reply, _ = await dispatch_gateway_message(
                cwd=tmp_path,
                session_store=sessions,
                scheduler_store=store,
                message=GatewayMessage(
                    id="query",
                    transport=transport,
                    user_id=user_id,
                    thread_id=thread_id,
                    text="runs",
                ),
            )
            assert "secret" not in json.dumps(reply.to_dict())
            assert [row["id"] for row in reply.data["runs"]] == ["run-public"]
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=sessions,
            scheduler_store=store,
            message=GatewayMessage(
                id="owner", transport="test", user_id="alice", thread_id="private", text="runs"
            ),
        )
        expected = {"run-public"} if local_only else {"run-public", "run-private"}
        assert {row["id"] for row in reply.data["runs"]} == expected
        assert "secret response" not in json.dumps(reply.to_dict())

        # Missing ownership evidence must never make an old private row public.
        (store.jobs_dir / job.id / "job.json").unlink()
        reply, _ = await dispatch_gateway_message(
            cwd=tmp_path,
            session_store=sessions,
            scheduler_store=store,
            message=GatewayMessage(
                id="orphan", transport="test", user_id="alice", thread_id="private", text="runs"
            ),
        )
        assert "secret" not in json.dumps(reply.to_dict())

    asyncio.run(check())


@pytest.mark.parametrize(
    "users",
    [
        ("alice.a", "alice-a"),
        ("Alice", "alice"),
        ("alice", " alice"),
        ("x" * 60 + "a", "x" * 60 + "b"),
    ],
)
def test_gateway_profiles_keep_colliding_legacy_names_separate(
    tmp_path: Path, users: tuple[str, str]
) -> None:
    store = GatewaySessionStore(root=tmp_path)
    first = store.get_or_create_profile(transport="test", user_id=users[0])
    store.save_profile(replace(first, metadata={"private": "first user's secret"}))
    second = store.get_or_create_profile(transport="test", user_id=users[1])
    assert first.id != second.id
    assert second.user_id == users[1]
    assert not second.metadata
    store.save_profile(replace(second, metadata={"private": "second user's secret"}))
    restarted = GatewaySessionStore(root=tmp_path)
    assert restarted.load_profile("test", users[0]).metadata == {"private": "first user's secret"}
    assert restarted.load_profile("test", users[1]).metadata == {"private": "second user's secret"}


@pytest.mark.parametrize("legacy_path", ["gwp-test-alice-a/profile.json", "test-alice-a.json"])
def test_gateway_profile_migration_checks_embedded_identity(
    tmp_path: Path, legacy_path: str
) -> None:
    store = GatewaySessionStore(root=tmp_path)
    legacy = GatewayUserProfile(
        id="gwp-test-alice-a",
        transport="test",
        user_id="alice.a",
        metadata={"private": "original secret"},
    )
    source = store.profiles_dir / legacy_path
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(legacy.to_dict()), encoding="utf-8")

    other = store.get_or_create_profile(transport="test", user_id="alice-a")
    assert not other.metadata
    original = store.load_profile("test", "alice.a")
    assert original.id != legacy.id
    assert original.metadata == legacy.metadata
    assert (store.profiles_dir / original.id / "profile.json").is_file()
    store.save_profile(replace(original, metadata={"private": "updated secret"}))
    assert json.loads(source.read_text())["metadata"] == legacy.metadata
    restarted = GatewaySessionStore(root=tmp_path)
    assert restarted.load_profile("test", "alice.a").metadata == {"private": "updated secret"}
    listed = {profile.user_id: profile for profile in restarted.list_profiles()}
    assert len(listed) == 2
    assert listed["alice.a"].metadata == {"private": "updated secret"}
    assert listed["alice-a"].metadata == {}
