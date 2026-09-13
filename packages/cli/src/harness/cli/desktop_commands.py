"""Native desktop window for an existing authenticated Harness server."""

from __future__ import annotations

import importlib
import ipaddress
import json
import os
from typing import Annotated
from urllib.parse import urlsplit

import typer


def desktop_command(
    url: Annotated[
        str, typer.Option("--url", help="Existing Harness server URL")
    ] = "http://127.0.0.1:8765",
    token_env: Annotated[
        str | None,
        typer.Option(
            "--token-env",
            help="Optional token environment reference; otherwise sign in inside the window",
        ),
    ] = None,
) -> None:
    """Open the native desktop client (install the CLI desktop extra first)."""
    parsed = urlsplit(url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise typer.BadParameter(
            "Use a root HTTP(S) server URL without credentials, query, or fragment"
        )
    if parsed.scheme == "http" and not loopback:
        raise typer.BadParameter("Remote desktop servers require HTTPS")
    token = os.environ.get(token_env) if token_env else None
    if token_env and not token:
        raise typer.BadParameter("The token environment reference is missing or empty")
    try:
        webview = importlib.import_module("webview")
    except ImportError:
        raise typer.BadParameter(
            "Native desktop support is not installed. Install the CLI desktop extra: uv sync --package cli --extra desktop"
        ) from None
    webview.settings.update(ALLOW_DOWNLOADS=True, ALLOW_FILE_URLS=False, IGNORE_SSL_ERRORS=False)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    window = webview.create_window(
        "Harness", origin + "/", width=1240, height=850, min_size=(700, 600)
    )
    if token:

        def connect() -> None:
            # Navigation must never inject a credential into an unrelated origin.
            window.run_js(
                f"if (window.location.origin === {json.dumps(origin)} && typeof window.harnessConnect === 'function') {{ window.harnessConnect({json.dumps(token)}); }}"
            )

        window.events.loaded += connect
    webview.start(private_mode=True, debug=False)


__all__ = ["desktop_command"]
