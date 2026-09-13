from __future__ import annotations

import importlib

import pytest

from harness.cli.server_builder import server_builder
from harness.cli.server_presentation import server_presentation
from harness.core import Capabilities, Done, Message, ToolRegistry
from harness.server import HarnessService, RunSubmission


class FakeAdapter:
    name = "configured"

    def __init__(self, calls, provider):
        self.calls = calls
        self.provider = provider

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        self.calls.append((self.provider, kwargs["model"]))
        yield Done(final_message=Message(role="assistant", content="selected model completed"))


def test_catalog_never_exposes_credentials_or_builds_desktop(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("""[default]
provider="ollama"
model="local-model"
[provider.openai]
model="chosen-openai-model"
api_key="never-publish-me"
[provider.private]
driver="openai-compatible"
api_key_env="PRIVATE_API_KEY"
base_url="https://example.invalid/v1"
[computer]
enabled=true
""")
    captured = []

    def tools(cwd, *, config, include):
        captured.append(config)
        assert not config.computer.enabled
        assert config.execution is None and config.browser is None
        assert config.mcp_servers == ()
        return ToolRegistry()

    monkeypatch.setattr("harness.cli.server_presentation._build_tools", tools)
    result = server_presentation(tmp_path, config, None, None)
    assert {item.id for item in result.providers} == {"ollama", "openai", "private"}
    serialized = result.model_dump_json()
    assert "never-publish-me" not in serialized and "PRIVATE_API_KEY" not in serialized
    assert "example.invalid" not in serialized
    assert result.default_model == "local-model" and len(captured) == 1
    clarification = next(tool for tool in result.tools if tool.name == "clarify")
    assert clarification.approval == "auto" and clarification.effect_scope == "session_ephemeral"
    assert clarification.parameters_schema is not None
    assert "questions" in clarification.parameters_schema["properties"]
    recall = next(tool for tool in result.tools if tool.name == "recall_memory")
    assert recall.approval == "auto" and recall.effect_scope == "read_only"
    assert recall.parameters_schema is not None
    assert recall.parameters_schema["properties"]["action"]["enum"] == ["list", "search", "get"]

    def unused(context):
        raise AssertionError("Catalog inspection must not construct an agent")

    service = HarnessService(tmp_path / "catalog.db", tmp_path, unused, presentation=result)
    assert not next(tool for tool in service.tool_catalog() if tool["name"] == "clarify")["exposed"]
    assert (
        next(tool for tool in service.tool_catalog() if tool["name"] == "recall_memory")[
            "effective_approval"
        ]
        == "deny"
    )


@pytest.mark.asyncio
async def test_concrete_server_builder_routes_selected_provider_and_model(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("""[default]
provider="ollama"
model="local-model"
[provider.openai]
model="chosen-openai-model"
[computer]
enabled=true
""")
    calls = []
    module = importlib.import_module("harness.cli.server_builder")
    monkeypatch.setattr(
        module, "_build_adapter", lambda provider, **kwargs: FakeAdapter(calls, provider)
    )

    def tools(cwd, *, config, include):
        assert not config.computer.enabled
        return ToolRegistry()

    monkeypatch.setattr(module, "_build_tools", tools)
    monkeypatch.setattr("harness.cli.server_presentation._build_tools", tools)
    service = HarnessService(
        tmp_path / "api.db",
        tmp_path,
        server_builder(tmp_path, config, None, None),
        presentation=server_presentation(tmp_path, config, None, None),
    )
    await service.start()
    try:
        run = await service.submit(
            "alice", RunSubmission(prompt="Use the selected provider", provider="openai")
        )
        async for _ in service.events("alice", run["id"]):
            pass
        assert (await service.store.run("alice", run["id"]))["state"] == "completed"
        assert calls == [("openai", "chosen-openai-model")]
    finally:
        await service.close()


def test_cli_worker_can_execute_provider_frozen_api_requests(tmp_path, monkeypatch):
    import asyncio

    import typer
    from typer.testing import CliRunner

    from harness.cli.batch_commands import make_batch_app
    from harness.server import ProviderOption, ServerPresentation

    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="openai"\nmodel="example"\n')
    calls = []

    def factory(workspace, config, provider, model):
        def build(context):
            from harness.core import Agent, FailoverPolicy

            return Agent(
                adapters={"openai": FakeAdapter(calls, "openai")},
                tools=ToolRegistry(),
                storage=context.storage,
                failover=FailoverPolicy(chain=["openai"]),
                default_model="example",
            )

        return build

    database = tmp_path / ".harness/server.db"

    async def queue():
        service = HarnessService(
            database,
            tmp_path,
            factory(tmp_path, config, None, None),
            presentation=ServerPresentation(
                providers=[ProviderOption(id="openai", default_model="example")],
                default_provider="openai",
                default_model="example",
            ),
        )
        await service.start(dispatch=False)
        try:
            return await service.submit_batch("local", [RunSubmission(prompt="queued by API")])
        finally:
            await service.close()

    batch = asyncio.run(queue())
    app = typer.Typer()
    app.add_typer(make_batch_app(factory), name="batch")
    result = CliRunner().invoke(
        app, ["batch", "--workspace", str(tmp_path), "--config", str(config), "work"]
    )
    assert result.exit_code == 0, result.output
    assert batch["id"] in result.output
    assert calls == [("openai", "example")]
