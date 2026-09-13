"""Bearer-only caller identity and explicit browser/host boundaries."""

from __future__ import annotations

import hmac
import ipaddress
import os
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send


class ServerAuth:
    def __init__(
        self,
        *,
        token_envs: Mapping[str, str],
        allowed_origins: Sequence[str] = (),
        allowed_hosts: Sequence[str] = ("localhost", "127.0.0.1", "::1"),
    ):
        if not token_envs:
            raise ValueError("Configure at least one user:token-environment reference")
        self.tokens: list[tuple[str, str]] = []
        for owner, name in token_envs.items():
            token = os.environ.get(name)
            if not owner.strip() or len(owner) > 512 or not token or len(token) < 32:
                raise ValueError(
                    f"Missing or invalid token reference for user {owner!r}; tokens must contain at least 32 characters"
                )
            if any(
                hmac.compare_digest(token.encode(), existing.encode())
                for _, existing in self.tokens
            ):
                raise ValueError("Each user must have a distinct bearer token")
            self.tokens.append((owner, token))
        self.allowed_origins = frozenset(allowed_origins)
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        if not self.allowed_hosts or "*" in self.allowed_hosts:
            raise ValueError("Explicit allowed hosts are required")
        for origin in allowed_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username
            ):
                raise ValueError("Allowed origins must be exact HTTP(S) origins without paths")

    def identity(self, authorization: str) -> str | None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return None
        owner = None
        for candidate, expected in self.tokens:
            if hmac.compare_digest(token.encode(), expected.encode()):
                owner = candidate
        return owner


class AuthMiddleware:
    def __init__(self, app: ASGIApp, auth: ServerAuth):
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        try:
            hostname = urlsplit("//" + host).hostname
        except ValueError:
            hostname = None
        if hostname not in self.auth.allowed_hosts:
            await JSONResponse({"detail": "Host is not allowed"}, status_code=400)(
                scope, receive, send
            )
            return
        origin = headers.get("origin")
        own_origin = f"{scope['scheme']}://{host}"
        if origin is not None and origin != own_origin and origin not in self.auth.allowed_origins:
            await JSONResponse({"detail": "Origin is not allowed"}, status_code=403)(
                scope, receive, send
            )
            return
        cors = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}
        if scope["method"] == "OPTIONS":
            await Response(
                status_code=204,
                headers={
                    **cors,
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, Last-Event-ID, MCP-Protocol-Version, MCP-Session-ID, A2A-Version",
                },
            )(scope, receive, send)
            return
        owner = self.auth.identity(headers.get("authorization", ""))
        static_asset = scope["method"] in ("GET", "HEAD") and scope["path"] in (
            "/",
            "/ui/app.js",
            "/ui/app.css",
        )
        if owner is None and not static_asset:
            await JSONResponse(
                {"detail": "Bearer authentication required"},
                status_code=401,
                headers={**cors, "WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return
        scope.setdefault("state", {})["owner"] = owner

        async def safe_send(message):
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers.update(
                    {
                        **cors,
                        "Cache-Control": "no-store",
                        "X-Content-Type-Options": "nosniff",
                        "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; media-src 'self' data:; frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
                    }
                )
            await send(message)

        await self.app(scope, receive, safe_send)


def validate_bind(
    host: str, *, allow_remote: bool, certfile: str | None, keyfile: str | None
) -> None:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback and (not allow_remote or not certfile or not keyfile):
        raise ValueError("Remote binding requires --allow-remote and both --tls-cert and --tls-key")
    if bool(certfile) != bool(keyfile):
        raise ValueError("Both --tls-cert and --tls-key are required together")
