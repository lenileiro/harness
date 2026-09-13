from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from harness.core import (
    Agent,
    Capabilities,
    Done,
    FailoverPolicy,
    Message,
    RunRequest,
    ToolRegistry,
)
from harness.core.schemas import ToolCall
from harness.core.tools import InboxApprovalHandler
from harness.storage.sqlite import SQLiteStorage
from harness.tools.homeassistant import (
    HomeAssistantConfig,
    HomeAssistantError,
    HomeAssistantToolset,
)


def config(**kwargs):
    return HomeAssistantConfig(
        enabled=True,
        base_url="http://home-assistant.invalid",
        entities=("light.desk", "sensor.office"),
        services=("light.turn_on",),
        service_fields={"light.turn_on": ("brightness",)},
        **kwargs,
    )


def tool(toolset, name):
    return next(value for value in toolset.tools if value.name == name)


@pytest.mark.asyncio
async def test_state_catalog_and_actions_are_scoped_and_only_one_token_is_forwarded():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer selected-token"
        assert "ambient-secret" not in str(request.headers)
        if request.url.path == "/api/states/light.desk":
            return httpx.Response(200, json={"entity_id": "light.desk", "state": "off"})
        if request.url.path == "/api/services":
            return httpx.Response(
                200,
                json=[
                    {
                        "domain": "light",
                        "services": {
                            "turn_on": {
                                "description": "Turn on a light",
                                "fields": {
                                    "brightness": {"example": 100},
                                    "ungranted": {"example": "not exposed"},
                                },
                            },
                            "turn_off": {},
                        },
                    },
                    {"domain": "lock", "services": {"unlock": {}}},
                ],
            )
        assert request.url.path == "/api/services/light/turn_on"
        assert json.loads(request.content) == {"entity_id": ["light.desk"], "brightness": 100}
        return httpx.Response(
            200,
            json=[
                {"entity_id": "light.desk", "state": "on"},
                {"entity_id": "sensor.private", "state": "sensitive"},
            ],
        )

    async with HomeAssistantToolset(
        config(),
        environment={"HOME_ASSISTANT_TOKEN": "selected-token", "OTHER_TOKEN": "ambient-secret"},
        transport=httpx.MockTransport(handle),
    ) as managed:
        listed = await tool(managed, "homeassistant_list_entities")(
            ToolCall(id="list", name="homeassistant_list_entities")
        )
        assert (
            json.loads(listed.content)["entities"] == ["light.desk", "sensor.office"]
            and requests == []
        )
        read = await tool(managed, "homeassistant_get_state")(
            ToolCall(
                id="read", name="homeassistant_get_state", arguments={"entity_id": "light.desk"}
            )
        )
        assert json.loads(read.content)["state"] == "off"
        catalog = await tool(managed, "homeassistant_list_services")(
            ToolCall(id="catalog", name="homeassistant_list_services")
        )
        assert (
            "turn_off" not in catalog.content
            and "unlock" not in catalog.content
            and "ungranted" not in catalog.content
        )
        call = await tool(managed, "homeassistant_call_service")(
            ToolCall(
                id="action",
                name="homeassistant_call_service",
                arguments={
                    "service": "light.turn_on",
                    "entity_ids": ["light.desk"],
                    "data": {"brightness": 100},
                },
            )
        )
        assert not call.is_error and "sensor.private" not in call.content
        assert len(requests) == 3
    assert managed.tools == []


@pytest.mark.asyncio
async def test_target_overrides_unallowed_entities_and_service_data_never_send():
    requests = []
    async with HomeAssistantToolset(
        config(),
        environment={"HOME_ASSISTANT_TOKEN": "test"},
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, json=[])
        ),
    ) as managed:
        action = tool(managed, "homeassistant_call_service")
        for arguments in (
            {"service": "light.turn_off", "entity_ids": ["light.desk"]},
            {"service": "light.turn_on", "entity_ids": ["light.other"]},
            {
                "service": "light.turn_on",
                "entity_ids": ["light.desk"],
                "data": {"entity_id": "all"},
            },
            {
                "service": "light.turn_on",
                "entity_ids": ["light.desk"],
                "data": {"target": {"area_id": "home"}},
            },
        ):
            assert (
                await action(ToolCall(id="deny", name=action.name, arguments=arguments))
            ).is_error
        assert requests == []
    with pytest.raises(ValueError):
        HomeAssistantConfig(
            enabled=True,
            base_url="http://home.invalid",
            services=("light.turn_on",),
            service_fields={"light.turn_on": ("entity_id",)},
        )


@pytest.mark.asyncio
async def test_redirects_and_ambiguous_failures_are_never_retried_or_echo_credentials():
    calls = []

    def fail(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-header-token", request=request)

    async with HomeAssistantToolset(
        config(), environment={"HOME_ASSISTANT_TOKEN": "test"}, transport=httpx.MockTransport(fail)
    ) as managed:
        action = tool(managed, "homeassistant_call_service")
        result = await action(
            ToolCall(
                id="action",
                name=action.name,
                arguments={"service": "light.turn_on", "entity_ids": ["light.desk"]},
            )
        )
        assert result.is_error and "may have reached" in result.content
        assert "private-header-token" not in result.content and len(calls) == 1
    calls.clear()

    def redirect(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://other.invalid/steal"})

    async with HomeAssistantToolset(
        config(),
        environment={"HOME_ASSISTANT_TOKEN": "test"},
        transport=httpx.MockTransport(redirect),
    ) as managed:
        result = await tool(managed, "homeassistant_get_state")(
            ToolCall(
                id="read", name="homeassistant_get_state", arguments={"entity_id": "light.desk"}
            )
        )
        assert result.is_error and len(calls) == 1


@pytest.mark.asyncio
async def test_bounded_response_and_cancellation_cleanup():
    async with HomeAssistantToolset(
        config(max_response_bytes=1024),
        environment={"HOME_ASSISTANT_TOKEN": "test"},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 2000)),
    ) as managed:
        result = await tool(managed, "homeassistant_get_state")(
            ToolCall(
                id="read", name="homeassistant_get_state", arguments={"entity_id": "light.desk"}
            )
        )
        assert result.is_error and "size limit" in result.content
    entered = asyncio.Event()

    async def wait(request):
        entered.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json={})

    manager = HomeAssistantToolset(
        config(), environment={"HOME_ASSISTANT_TOKEN": "test"}, transport=httpx.MockTransport(wait)
    )

    async def run():
        async with manager as managed:
            await tool(managed, "homeassistant_get_state")(
                ToolCall(
                    id="read", name="homeassistant_get_state", arguments={"entity_id": "light.desk"}
                )
            )

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager._client is None


@pytest.mark.asyncio
async def test_actual_agent_pauses_before_home_service_request(tmp_path):
    calls = []

    class Adapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(tool_use=True)

        async def stream(self, **kwargs):
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="ha-action",
                            name="homeassistant_call_service",
                            arguments={"service": "light.turn_on", "entity_ids": ["light.desk"]},
                        )
                    ],
                )
            )

        async def cancel(self, session_id):
            pass

    async with HomeAssistantToolset(
        config(),
        environment={"HOME_ASSISTANT_TOKEN": "test"},
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, json=[])
        ),
    ) as managed:
        registry = ToolRegistry()
        for entry in managed.tools:
            registry.register(entry)
        storage = SQLiteStorage(path=tmp_path / "sessions.db")
        try:
            agent = Agent(
                adapters={"fake": Adapter()},
                tools=registry,
                storage=storage,
                failover=FailoverPolicy(chain=["fake"]),
                default_model="test",
                default_cwd=str(tmp_path),
                approval_store=storage,
                approval_handler=InboxApprovalHandler(approval_store=storage),
                pause_on_approval=True,
            )
            events = [
                event async for event in agent.run(RunRequest(prompt="Turn on the desk light"))
            ]
            assert calls == []
            assert any(
                isinstance(event, Done)
                and event.structured_result == {"status": "waiting_for_approval"}
                for event in events
            )
            approvals = await storage.list_approvals(status="pending")
            assert len(approvals) == 1 and approvals[0].tool_name == "homeassistant_call_service"
        finally:
            await storage.close()


@pytest.mark.asyncio
async def test_disabled_and_missing_or_invalid_credentials():
    async with HomeAssistantToolset(HomeAssistantConfig()) as disabled:
        assert disabled.tools == []
    for environment in ({}, {"HOME_ASSISTANT_TOKEN": "private-ünicode-secret"}):
        with pytest.raises(HomeAssistantError, match="missing or invalid"):
            async with HomeAssistantToolset(config(), environment=environment):
                pytest.fail("must not open")
