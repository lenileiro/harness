"""Explicitly allowlisted presentation data for authenticated web clients."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from harness.cli.common import KNOWN_PROVIDERS, _build_tools
from harness.cli.config import load_config
from harness.core.clarification import ClarifyArguments
from harness.core.tools_clarify import ClarifyTool
from harness.core.tools_durable_memory import RecallMemoryArguments, RecallMemoryTool
from harness.server.models import ProviderOption, ServerPresentation, ToolPresentation
from harness.tools.computer import ComputerConfig
from harness.tools.media import MediaConfig

SAFE_SERVER_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "web_search",
    "fetch_url",
}


def server_presentation(
    workspace: Path, config_path: Path | None, provider: str | None, model: str | None
) -> ServerPresentation:
    try:
        config = load_config(config_path)
    except ValueError:
        raise ValueError("Invalid server configuration; inspect its syntax locally") from None
    selected = provider or config.default_provider or "ollama"
    providers = {selected}
    providers.update(
        name
        for name, settings in config.provider_settings.items()
        if name in KNOWN_PROVIDERS or settings.get("driver") == "openai-compatible"
    )
    if "codex" in providers and config.provider("codex").get("mode", "exec") != "app-server":
        providers.remove("codex")
    selected_model = (
        model or config.provider(selected).get("model") or config.default_model or "llama3.2"
    )
    options = [
        ProviderOption(
            id=name,
            default_model=selected_model
            if name == selected
            else config.provider(name).get("model"),
        )
        for name in sorted(providers)
    ]
    # Match the server builder's enabled surface; managed connectors and local
    # execution configuration/headers/environment values are never serialized.
    safe_config = replace(
        config,
        execution=None,
        browser=None,
        computer=ComputerConfig(),
        media=MediaConfig(),
        mcp_servers=(),
        skills_enabled=False,
        delegation_enabled=False,
    )
    registry = _build_tools(workspace, config=safe_config, include=SAFE_SERVER_TOOLS)
    tools = [
        ToolPresentation(
            name=RecallMemoryTool.name,
            description=RecallMemoryTool.description,
            parameters_schema=RecallMemoryArguments.model_json_schema(),
            approval=config.approval.get(RecallMemoryTool.name, RecallMemoryTool.approval),
            effect_scope=RecallMemoryTool.effect_scope,
        ),
        ToolPresentation(
            name="clarify",
            description=ClarifyTool.description,
            parameters_schema=ClarifyArguments.model_json_schema(),
            approval=config.approval.get("clarify", ClarifyTool.approval),
            effect_scope=ClarifyTool.effect_scope,
        ),
    ]
    for name in registry.names():
        tool = registry.get(name)
        tools.append(
            ToolPresentation(
                name=tool.name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
                approval=config.approval.get(name, tool.approval),
                effect_scope=getattr(tool, "effect_scope", "unknown"),
            )
        )
    return ServerPresentation(
        providers=options,
        default_provider=selected if selected in providers else None,
        default_model=selected_model,
        tools=tools,
    )
