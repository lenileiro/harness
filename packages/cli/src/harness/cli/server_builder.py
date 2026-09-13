"""Construct server agents without granting conversational callers host execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from harness.cli.common import (
    _build_adapter,
    _build_tools,
    _load_cli_config,
    _resolve_chain,
    console,
)
from harness.cli.gateway_tool_boundary import install_gateway_tool_boundary
from harness.cli.runtime_agent import build_agent
from harness.cli.runtime_helpers import build_search_fn
from harness.core import Agent, ConfigurationError
from harness.server import AgentBuilder, RunContext


def server_builder(
    workspace: Path, config_path: Path | None, provider: str | None, model: str | None
) -> AgentBuilder:
    config = _load_cli_config(config_path)
    chain = _resolve_chain(failover_flag=None, provider_flag=provider, config=config)
    if "codex" in chain and config.provider("codex").get("mode", "exec") != "app-server":
        raise ConfigurationError("API serving requires a provider that dispatches Harness tools")
    # Shared-workspace servers expose explicitly allowlisted path tools through
    # the remote boundary. Shell/native execution and shared managed credentials
    # require a separate process/workspace identity rather than a chat allowlist.
    from harness.tools.computer import ComputerConfig
    from harness.tools.media import MediaConfig

    config = replace(
        config,
        execution=None,
        browser=None,
        computer=ComputerConfig(),
        media=MediaConfig(),
        mcp_servers=(),
        skills_enabled=False,
        delegation_enabled=False,
    )

    def build(context: RunContext) -> Agent:
        run_workspace = context.workspace
        run_chain = [context.provider] if context.provider else chain
        if "codex" in run_chain and config.provider("codex").get("mode", "exec") != "app-server":
            raise ConfigurationError(
                "API serving requires a provider that dispatches Harness tools"
            )
        run_model = (
            context.model
            or (model if run_chain == chain else None)
            or config.provider(run_chain[0]).get("model")
            or config.default_model
            or "llama3.2"
        )
        agent = build_agent(
            chain=run_chain,
            base_url=None,
            model=run_model,
            storage=context.storage,
            cwd=run_workspace,
            config=config,
            yes=False,
            build_adapter=_build_adapter,
            build_tools=lambda cwd: _build_tools(
                cwd,
                config=config,
                include={
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_dir",
                    "glob",
                    "web_search",
                    "fetch_url",
                },
            ),
            build_search_fn=build_search_fn,
            console=console,
            inbox=True,
            pause_on_approval=True,
            approval_store=context.storage,
            activity_store=context.storage,
            memory_store=context.storage,
            auxiliary_tools_enabled=False,
            project_context_enabled=False,
            skip_builtin_verify_before_done=True,
        )
        install_gateway_tool_boundary(agent, run_workspace)
        return agent

    return build
