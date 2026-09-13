"""Explicit MCP server configuration; secret references resolve only at connection time."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.core.schemas import ApprovalDecision


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=40)
    transport: Literal["stdio", "streamable-http"] = "stdio"
    enabled: bool = True
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict, repr=False)
    env_from: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    headers_from: dict[str, str] = Field(default_factory=dict)
    bearer_token_env: str | None = None
    oauth: bool = False
    oauth_scopes: str | None = None
    oauth_redirect_port: int = Field(default=8766, ge=1024, le=65535)
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()
    expose_resources: bool = False
    expose_prompts: bool = False
    approval: ApprovalDecision = "prompt"
    timeout: float = Field(default=30.0, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_transport(self) -> Self:
        if self.transport == "stdio":
            if not self.command or not self.command.strip():
                raise ValueError("stdio MCP servers require a command")
            if self.url or self.headers or self.headers_from or self.bearer_token_env or self.oauth:
                raise ValueError("URL and headers are only valid for streamable-http servers")
        else:
            import httpx

            try:
                url = httpx.URL(self.url or "")
            except httpx.InvalidURL:
                raise ValueError(
                    "streamable-http MCP servers require a valid HTTP(S) URL"
                ) from None
            if url.scheme not in {"http", "https"} or not url.host or url.fragment:
                raise ValueError("streamable-http MCP servers require a valid HTTP(S) URL")
            if url.username or url.password:
                raise ValueError("use bearer_token_env or headers_from instead of URL credentials")
            if (
                self.oauth
                and url.scheme != "https"
                and url.host not in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError("MCP OAuth requires HTTPS except for loopback servers")
            if self.command or self.args or self.cwd or self.env or self.env_from:
                raise ValueError("command, args, cwd and env are only valid for stdio servers")
        if set(self.env) & set(self.env_from):
            raise ValueError("env and env_from must not define the same variable")
        headers = [name.lower() for name in (*self.headers, *self.headers_from)]
        if len(headers) != len(set(headers)):
            raise ValueError("header names must be unique, ignoring case")
        if self.bearer_token_env and "authorization" in headers:
            raise ValueError(
                "bearer_token_env and an Authorization header cannot both be configured"
            )
        if self.oauth and (self.bearer_token_env or "authorization" in headers):
            raise ValueError("OAuth cannot be combined with explicit Authorization credentials")
        if any(not name or not reference for name, reference in self.env_from.items()):
            raise ValueError("env_from requires nonempty variable names and references")
        if any(not name or not reference for name, reference in self.headers_from.items()):
            raise ValueError("headers_from requires nonempty header names and references")
        return self

    def resolve_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Only explicit variables plus the SDK's minimal OS environment are inherited."""
        source = os.environ if environ is None else environ
        return {**self.env, **_resolve(self.env_from, source, self.name)}

    def resolve_headers(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        source = os.environ if environ is None else environ
        headers = {**self.headers, **_resolve(self.headers_from, source, self.name)}
        if self.bearer_token_env:
            token = _resolve({"Authorization": self.bearer_token_env}, source, self.name)
            headers["Authorization"] = f"Bearer {token['Authorization']}"
        return headers


def _resolve(
    references: Mapping[str, str], environ: Mapping[str, str], server: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, reference in references.items():
        value = environ.get(reference)
        if not value:
            raise ValueError(f"MCP server {server!r} requires environment variable {reference!r}")
        result[name] = value
    return result


def parse_mcp_servers(raw: object) -> tuple[MCPServerConfig, ...]:
    """Parse either ``[mcp.servers.NAME]`` tables or a list of named server tables."""
    if raw is None:
        return ()
    if isinstance(raw, dict):
        rows = []
        for name, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError("each MCP server must be a configuration table")
            if "name" in value and value["name"] != name:
                raise ValueError("MCP server name must match its table key")
            rows.append({**value, "name": name})
    elif isinstance(raw, list | tuple):
        rows = list(raw)
    else:
        raise ValueError("mcp.servers must be a table or list of server configurations")
    configs = tuple(MCPServerConfig.model_validate(row) for row in rows)
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("MCP server names must be unique")
    return configs
