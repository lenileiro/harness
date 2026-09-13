"""Server entrypoint; caller supplies the application's configured Agent builder."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from harness.cli.server_presentation import server_presentation
from harness.server import AgentBuilder, HarnessService, ServerAuth, create_app, validate_bind
from harness.server.a2a_callbacks import CallbackGrant

BuilderFactory = Callable[[Path, Path | None, str | None, str | None], AgentBuilder]


def _references(values: list[str]) -> dict[str, str]:
    references = {}
    for value in values:
        owner, separator, variable = value.partition(":")
        if not separator or not owner.strip() or not variable.strip() or owner in references:
            raise ValueError("Use distinct --token-env USER:ENVIRONMENT_VARIABLE references")
        references[owner] = variable
    return references


def make_serve_command(builder_factory: BuilderFactory):
    def serve_command(
        workspace: Annotated[
            Path, typer.Option("--workspace", "-C", help="Workspace for all server executions")
        ] = Path("."),
        config: Annotated[Path | None, typer.Option("--config")] = None,
        adapter: Annotated[str | None, typer.Option("--adapter")] = None,
        model: Annotated[str | None, typer.Option("--model")] = None,
        database: Annotated[
            Path | None,
            typer.Option("--database", help="Persistent session and queue SQLite database"),
        ] = None,
        token_env: Annotated[
            list[str] | None,
            typer.Option(
                "--token-env", help="Caller identity USER:ENV_VAR; repeat for multiple callers"
            ),
        ] = None,
        host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8765,
        allow_remote: Annotated[
            bool, typer.Option("--allow-remote", help="Require TLS for binding beyond loopback")
        ] = False,
        tls_cert: Annotated[Path | None, typer.Option("--tls-cert")] = None,
        tls_key: Annotated[Path | None, typer.Option("--tls-key")] = None,
        allowed_host: Annotated[
            list[str] | None,
            typer.Option(
                "--allowed-host", help="Explicit additional HTTP host names; never wildcards"
            ),
        ] = None,
        allowed_origin: Annotated[
            list[str] | None,
            typer.Option("--allowed-origin", help="Explicit additional browser origins"),
        ] = None,
        expose_tool: Annotated[
            list[str] | None,
            typer.Option(
                "--expose-tool", help="Tool names allowed through Harness gates; defaults to none"
            ),
        ] = None,
        mcp: Annotated[
            bool, typer.Option("--mcp", help="Mount the official MCP HTTP transport at /mcp/")
        ] = False,
        mcp_operation: Annotated[
            list[str] | None,
            typer.Option(
                "--mcp-operation",
                help="MCP service methods to expose; defaults to read-only methods",
            ),
        ] = None,
        a2a_url: Annotated[
            str | None,
            typer.Option(
                "--a2a-url",
                help="Enable authenticated A2A at this explicit public HTTP(S) URL ending /a2a",
            ),
        ] = None,
        a2a_callbacks: Annotated[
            Path | None,
            typer.Option(
                "--a2a-callbacks",
                help="JSON list of explicit owner, HTTPS URL and signing-secret environment grants",
            ),
        ] = None,
        workers: Annotated[
            int,
            typer.Option(
                "--workers",
                min=1,
                max=32,
                help="Concurrent queued runs, within a single owning server process",
            ),
        ] = 2,
    ) -> None:
        """Serve authenticated runs, sessions, approvals, batches, and exports."""
        resolved_workspace = workspace.expanduser().resolve()
        try:
            if not resolved_workspace.is_dir():
                raise ValueError("Workspace must be an existing directory")
            certfile = str(tls_cert.expanduser().resolve()) if tls_cert else None
            keyfile = str(tls_key.expanduser().resolve()) if tls_key else None
            validate_bind(host, allow_remote=allow_remote, certfile=certfile, keyfile=keyfile)
            for path in (tls_cert, tls_key):
                if path is not None and not path.expanduser().is_file():
                    raise ValueError("TLS certificate and key must be existing files")
            hosts = ["localhost", "127.0.0.1", "::1", *(allowed_host or [])]
            if host not in ("0.0.0.0", "::"):
                hosts.append(host)
            auth = ServerAuth(
                token_envs=_references(token_env or []),
                allowed_hosts=hosts,
                allowed_origins=allowed_origin or [],
            )
            service = HarnessService(
                database or resolved_workspace / ".harness" / "server.db",
                resolved_workspace,
                builder_factory(resolved_workspace, config, adapter, model),
                exposed_tools=expose_tool or [],
                max_workers=workers,
                presentation=server_presentation(resolved_workspace, config, adapter, model),
            )
            callback_grants = None
            if a2a_callbacks is not None:
                try:
                    with a2a_callbacks.expanduser().open("rb") as stream:
                        raw_grants = stream.read(64 * 1024 + 1)
                    if len(raw_grants) > 64 * 1024:
                        raise ValueError
                    values = json.loads(raw_grants)
                    if not isinstance(values, list) or not values:
                        raise ValueError
                    callback_grants = [CallbackGrant.model_validate(value) for value in values]
                except (OSError, ValueError):
                    raise ValueError(
                        "Invalid A2A callback grant file; use a JSON list (at most 64 KiB) of owner, url, secret_env and optional bearer_env"
                    ) from None
            app = create_app(
                service,
                auth=auth,
                expose_mcp=mcp,
                a2a_url=a2a_url,
                a2a_callback_grants=callback_grants,
                mcp_operations=tuple(mcp_operation)
                if mcp_operation
                else ("run_status", "sessions", "messages"),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
        typer.echo(
            f"Harness API: {'https' if certfile else 'http'}://{host}:{port}/v1 (bearer authentication required)"
        )
        uvicorn.run(
            app, host=host, port=port, ssl_certfile=certfile, ssl_keyfile=keyfile, access_log=False
        )

    return serve_command


__all__ = ["BuilderFactory", "make_serve_command"]
