from dataclasses import replace
from typing import ClassVar

import pytest
from rich.console import Console

from harness.cli.a2a_tools import A2AConfig
from harness.cli.common import _build_tools
from harness.cli.config import HarnessConfig, load_config
from harness.cli.honcho_tools import HonchoConfig
from harness.cli.portal_tools import PortalConfig
from harness.cli.runtime_agent import build_agent
from harness.core import Capabilities, Done, Message, RunRequest, ToolResult
from harness.storage.memory import InMemoryStorage
from harness.tools.computer import ComputerConfig


class Adapter:
    name = "test"

    def __init__(self):
        self.calls = []

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield Done(final_message=Message(role="assistant", content="Complete"))


class Tool:
    description = "Integration fixture"
    parameters_schema: ClassVar[dict] = {"type": "object", "properties": {}}
    approval = "prompt"
    effect_scope = "external_side_effect"

    def __init__(self, name):
        self.name = name

    async def __call__(self, call):
        return ToolResult(tool_call_id=call.id, name=self.name, content="fixture")


@pytest.mark.asyncio
async def test_local_managed_connectors_compose_with_durable_children_and_close(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    opened, closed, bound = [], [], []

    def toolset(name, tools):
        class Owner:
            def __init__(self, *args, **kwargs):
                self.tools = [Tool(item) for item in tools]

            async def __aenter__(self):
                opened.append(name)
                return self

            async def __aexit__(self, *args):
                closed.append(name)

            def bind(self, session):
                bound.append((name, session.id))
                return [Tool("honcho_sync" if name == "honcho" else "a2a_call")]

        return Owner

    monkeypatch.setattr("harness.tools.computer.ComputerToolset", toolset("computer", ["computer"]))
    monkeypatch.setattr(
        "harness.cli.portal_tools.PortalToolset", toolset("portal", ["web_search", "fetch_url"])
    )
    monkeypatch.setattr("harness.cli.honcho_tools.HonchoToolset", toolset("honcho", []))
    monkeypatch.setattr("harness.cli.a2a_tools.A2AToolset", toolset("a2a", []))
    config = HarnessConfig(
        a2a=A2AConfig(enabled=True),
        computer=ComputerConfig(enabled=True),
        honcho=HonchoConfig(enabled=True),
        portal=PortalConfig(enabled=True, routes=("web",)),
        delegation_enabled=True,
        provider_settings={
            "nous": {
                "base_url": "https://model.example/v1",
                "oauth": {
                    "device_authorization_endpoint": "https://auth.example/device",
                    "token_endpoint": "https://auth.example/token",
                    "client_id": "test",
                },
            }
        },
    )
    adapter = Adapter()
    storage = InMemoryStorage()
    agent = build_agent(
        chain=["test"],
        base_url=None,
        model="test",
        storage=storage,
        cwd=tmp_path,
        config=config,
        yes=False,
        build_adapter=lambda *args, **kwargs: adapter,
        build_tools=lambda cwd: _build_tools(cwd, config=config),
        build_search_fn=lambda: None,
        console=Console(quiet=True),
        memory_store=storage,
        activity_store=storage,
        skip_builtin_verify_before_done=True,
    )
    async with agent:
        for identifier in ["first", "second"]:
            async for _ in agent.run(
                RunRequest(prompt="Work", session_id=identifier, model="test")
            ):
                pass
            names = set(agent.tools.names())
            assert {
                "computer",
                "honcho_sync",
                "delegate",
                "web_search",
                "fetch_url",
                "a2a_call",
            } <= names
            assert agent.tools.get("computer").approval == "prompt"
        assert opened == ["computer", "portal", "honcho", "a2a"] and closed == []
    assert closed == ["a2a", "honcho", "portal", "computer"]
    assert bound == [
        (name, session) for session in ("first", "second") for name in ("honcho", "a2a")
    ]
    assert len(adapter.calls) == 2
    # The same configured desktop cannot enter an auxiliary-disabled runtime.
    with pytest.raises(Exception, match="Computer control"):
        build_agent(
            chain=["test"],
            base_url=None,
            model="test",
            storage=storage,
            cwd=tmp_path,
            config=replace(config, delegation_enabled=False),
            yes=False,
            build_adapter=lambda *args, **kwargs: adapter,
            build_tools=lambda cwd: _build_tools(cwd, config=config),
            build_search_fn=lambda: None,
            console=Console(quiet=True),
            auxiliary_tools_enabled=False,
        )


def test_configuration_loads_explicit_integrations_without_opening_clients(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[computer]\nenabled=true\n[honcho]\nenabled=true\n[portal]\nenabled=true\nroutes=["web"]\n'
    )
    config = load_config(path)
    assert config.computer.enabled and config.honcho.enabled
    assert config.portal.tool_names() == {"web_search", "fetch_url"}
