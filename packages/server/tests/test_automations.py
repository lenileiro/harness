from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from harness.core import Agent, FailoverPolicy, ToolRegistry
from harness.core.schemas import Message, ToolCall
from harness.server import HarnessService, ServiceError
from harness.server.automations import ScheduleSubmission
from harness.server.models import (
    ProviderOption,
    ServerPresentation,
    ToolPresentation,
    UserPreferences,
)

from .test_service import FakeAdapter, RecordingTool, bob_headers, builder, client_for, terminal


def presentation():
    return ServerPresentation(
        providers=[
            ProviderOption(id="fake", default_model="first-model"),
            ProviderOption(id="other", default_model="other-model"),
        ],
        default_provider="fake",
        default_model="first-model",
        tools=[
            ToolPresentation(
                name="read",
                description="Read source",
                parameters_schema={"type": "object"},
                approval="auto",
                effect_scope="read_only",
            ),
            ToolPresentation(
                name="write",
                parameters_schema={"type": "object"},
                approval="auto",
                effect_scope="workspace_durable",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_preferences_configuration_and_tools_are_scoped_and_provider_is_enforced(
    tmp_path, monkeypatch
):
    contexts = []
    adapter = FakeAdapter()

    def build(context):
        contexts.append(context)
        name = context.provider or "fake"
        return Agent(
            adapters={name: adapter},
            tools=ToolRegistry(),
            storage=context.storage,
            failover=FailoverPolicy(chain=[name]),
            default_model=context.model or "test",
        )

    service = HarnessService(
        tmp_path / "api.db",
        tmp_path,
        build,
        presentation=presentation(),
        exposed_tools=["read", "write"],
    )
    async with client_for(service, monkeypatch) as client:
        config = (await client.get("/v1/configuration")).json()
        assert config["default_provider"] == "fake" and "api_key" not in json.dumps(config)
        tools = (await client.get("/v1/tools")).json()["tools"]
        assert {item["name"]: item["effective_approval"] for item in tools} == {
            "read": "auto",
            "write": "prompt",
        }
        assert (
            await client.post(
                "/v1/preferences",
                json={"provider": "other", "model": "chosen", "timezone": "Europe/Tallinn"},
            )
        ).status_code == 200
        assert (await client.get("/v1/preferences", headers=bob_headers())).json()[
            "provider"
        ] is None
        first = (await client.post("/v1/runs", json={"prompt": "first"})).json()
        assert (await terminal(client, first["id"]))["state"] == "completed"
        assert contexts[0].provider == "other" and contexts[0].model == "chosen"
        await client.post("/v1/preferences", json={"provider": "fake", "model": "changed"})
        follow = (
            await client.post(
                "/v1/runs", json={"prompt": "continue", "session_id": first["session_id"]}
            )
        ).json()
        assert (await terminal(client, follow["id"]))["state"] == "completed"
        assert contexts[-1].provider == "other" and contexts[-1].model == "chosen"
        rejected = await client.post("/v1/runs", json={"prompt": "forbidden", "provider": "codex"})
        assert rejected.status_code == 403
        assert (
            await client.post(
                "/v1/preferences", json={"provider": "fake", "api_key": "not-accepted"}
            )
        ).status_code == 422
        assert "not-accepted" not in (await client.get("/v1/preferences")).text
    reloaded = HarnessService(tmp_path / "api.db", tmp_path, build, presentation=presentation())
    await reloaded.start(dispatch=False)
    try:
        assert (await reloaded.preferences("alice")).model == "changed"
        assert (await reloaded.preferences("bob")).model is None
    finally:
        await reloaded.close()


@pytest.mark.asyncio
async def test_atomic_occurrence_queueing_restart_and_owner_access(tmp_path):
    path = tmp_path / "api.db"
    service = HarnessService(path, tmp_path, builder(FakeAdapter()))
    await service.start(dispatch=False)
    try:
        scheduled = await service.automations.create(
            "alice",
            ScheduleSubmission(
                title="Daily note", prompt="Write a note", every="1m", timezone="Europe/Tallinn"
            ),
        )
        moment = datetime.now(UTC) + timedelta(minutes=2)
        first, second = await asyncio.gather(
            service.automations.tick(moment=moment), service.automations.tick(moment=moment)
        )
        assert len(first) + len(second) == 1
        rows = await service.store.rows("SELECT * FROM api_schedule_runs")
        assert len(rows) == 1
        with pytest.raises(ServiceError) as error:
            await service.automations.get("bob", scheduled["id"])
        assert error.value.status == 404
        assert await service.automations.tick(moment=moment + timedelta(hours=1)) == []
    finally:
        await service.close()
    restarted = HarnessService(path, tmp_path, builder(FakeAdapter()))
    await restarted.start(dispatch=False)
    try:
        assert len(await restarted.store.rows("SELECT id FROM api_runs")) == 1
        assert await restarted.automations.tick(moment=moment + timedelta(hours=1)) == []
        paused = await restarted.automations.change("alice", scheduled["id"], "pause")
        assert paused["state"] == "paused"
        assert await restarted.automations.tick(moment=moment + timedelta(hours=1)) == []
        resumed = await restarted.automations.change("alice", scheduled["id"], "resume")
        assert resumed["state"] == "active"
        cancelled = await restarted.automations.change("alice", scheduled["id"], "cancel")
        assert (
            cancelled["state"] == "cancelled"
            and cancelled["runs"][0]["run"]["state"] == "cancelled"
        )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_occurrence_rolls_back_with_queue_failure(tmp_path, monkeypatch):
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(FakeAdapter()))
    await service.start(dispatch=False)
    try:
        schedule = await service.automations.create(
            "alice",
            ScheduleSubmission(
                title="Once",
                prompt="test",
                at=(datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            ),
        )
        assert service.store.db is not None
        await service.store.db.execute(
            "CREATE TRIGGER refuse_run BEFORE INSERT ON api_runs BEGIN SELECT RAISE(ABORT,'offline test'); END"
        )
        await service.store.db.commit()
        assert await service.automations.tick(moment=datetime.now(UTC) + timedelta(minutes=2)) == []
        unchanged = await service.automations.get("alice", schedule["id"])
        assert (
            unchanged["state"] == "active" and unchanged["next_run_at"] == schedule["next_run_at"]
        )
        assert await service.store.rows("SELECT * FROM api_schedule_runs") == []
        assert await service.store.rows("SELECT * FROM api_sessions") == []
    finally:
        await service.close()


class ToolAdapter(FakeAdapter):
    async def stream(self, *, messages, **kwargs):
        from harness.core.events import Done

        if not any(message.role == "tool" for message in messages):
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="scheduled-action", name="record", arguments={"text": "scheduled"}
                        )
                    ],
                )
            )
        else:
            yield Done(final_message=Message(role="assistant", content="finished"))


@pytest.mark.asyncio
async def test_http_scheduled_prompt_reuses_approval_pipeline_and_waits_for_paused_occurrence(
    tmp_path, monkeypatch
):
    tool = RecordingTool()
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(ToolAdapter(), tool), exposed_tools=["record"]
    )
    async with client_for(service, monkeypatch) as client:
        created = await client.post(
            "/v1/schedules",
            json={"title": "Reviewed action", "prompt": "record value", "every": "60s"},
        )
        assert created.status_code == 201, created.text
        schedule = created.json()
        future = datetime.now(UTC) + timedelta(minutes=2)
        run_id = (await service.automations.tick(moment=future))[0]
        assert (await terminal(client, run_id))["state"] == "paused"
        assert tool.values == []
        assert await service.automations.tick(moment=future + timedelta(minutes=3)) == []
        approval = (await client.get("/v1/approvals")).json()["approvals"][0]
        assert (
            await client.post(f"/v1/approvals/{approval['id']}/resolve", json={"granted": True})
        ).status_code == 200
        resumed = (await client.post(f"/v1/runs/{run_id}/resume", json={})).json()
        assert (await terminal(client, resumed["id"]))["state"] == "completed"
        assert tool.values == ["scheduled"]
        updated = (await client.get(f"/v1/schedules/{schedule['id']}")).json()
        assert updated["runs"][0]["run"]["id"] == resumed["id"]
        assert len(await service.automations.tick(moment=future + timedelta(minutes=3))) == 1
        assert (
            await client.post(
                f"/v1/schedules/{schedule['id']}/pause", json={}, headers=bob_headers()
            )
        ).status_code == 404
        assert (
            await client.post(f"/v1/schedules/{schedule['id']}/cancel", json={})
        ).status_code == 200


@pytest.mark.asyncio
async def test_schedule_timezone_validation_frozen_defaults_and_unadvertised_provider(tmp_path):
    service = HarnessService(
        tmp_path / "api.db", tmp_path, builder(FakeAdapter()), presentation=presentation()
    )
    await service.start(dispatch=False)
    try:
        await service.save_preferences(
            "alice",
            UserPreferences(provider="other", model="scheduled-model", timezone="Europe/Tallinn"),
        )
        schedule = await service.automations.create(
            "alice", ScheduleSubmission(title="Local noon", prompt="check", cron="0 12 * * 1-5")
        )
        assert schedule["schedule"]["timezone"] == "Europe/Tallinn"
        assert schedule["request"]["provider"] == "other"
        await service.save_preferences("alice", UserPreferences(provider="fake", model="changed"))
        assert (await service.automations.get("alice", schedule["id"]))["request"][
            "model"
        ] == "scheduled-model"
        for invalid in (
            {"every": "1s"},
            {"cron": "broken"},
            {"every": "1m", "timezone": "Invalid/Zone"},
            {"at": "2020-01-01"},
            {"at": "2027-03-28T03:30:00", "timezone": "Europe/Tallinn"},
            {"every": "1m", "provider": "unadvertised"},
        ):
            with pytest.raises(ServiceError):
                await service.automations.create(
                    "alice",
                    ScheduleSubmission.model_validate(
                        {"title": "invalid", "prompt": "x", **invalid}
                    ),
                )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_one_time_fired_schedule_can_cancel_its_running_work(tmp_path):
    adapter = FakeAdapter(block=asyncio.Event())
    service = HarnessService(tmp_path / "api.db", tmp_path, builder(adapter))
    await service.start()
    try:
        schedule = await service.automations.create(
            "alice",
            ScheduleSubmission(
                title="one time",
                prompt="work",
                at=(datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            ),
        )
        run_id = (await service.automations.tick(moment=datetime.now(UTC) + timedelta(minutes=2)))[
            0
        ]
        await asyncio.wait_for(adapter.entered.wait(), 3)
        assert (await service.automations.get("alice", schedule["id"]))["state"] == "completed"
        cancelled = await service.automations.change("alice", schedule["id"], "cancel")
        assert cancelled["state"] == "cancelled"
        assert (await service.store.run("alice", run_id))["state"] == "cancelled"
        assert adapter.cancelled.is_set()
    finally:
        await service.close()
