"""Full official-SDK discovery/PKCE/refresh contract, with an offline HTTP server."""

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from harness.tools.mcp.config import MCPServerConfig
from harness.tools.mcp.oauth import FileTokenStorage, oauth_provider


@pytest.mark.asyncio
async def test_sdk_device_browser_exchange_state_pkce_and_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    server = MCPServerConfig(
        name="example", transport="streamable-http", url="https://resource.example/mcp", oauth=True
    )
    assert server.url is not None
    authorization = {}
    grants = []
    registrations = []

    async def redirect(url):
        authorization.update(parse_qs(urlparse(url).query))

    async def callback():
        return "authorization-code", authorization["state"][0]

    def respond(request):
        path = request.url.path
        if path == "/mcp":
            if request.headers.get("authorization") in {
                "Bearer first-access",
                "Bearer refreshed-access",
            }:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://resource.example/.well-known/oauth-protected-resource"'
                },
            )
        if path == "/.well-known/oauth-protected-resource":
            return httpx.Response(
                200,
                json={
                    "resource": "https://resource.example/mcp",
                    "authorization_servers": ["https://auth.example"],
                    "scopes_supported": ["tools"],
                },
            )
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example",
                    "authorization_endpoint": "https://auth.example/authorize",
                    "token_endpoint": "https://auth.example/token",
                    "registration_endpoint": "https://auth.example/register",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["none"],
                },
            )
        if path == "/register":
            import json

            registrations.append(json.loads(request.content))
            return httpx.Response(201, json={**registrations[-1], "client_id": "registered-client"})
        if path == "/token":
            form = parse_qs(request.content.decode())
            grants.append(form)
            if form["grant_type"] == ["authorization_code"]:
                challenge = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(form["code_verifier"][0].encode()).digest()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                assert challenge == authorization["code_challenge"][0]
                assert form["code"] == ["authorization-code"]
                return httpx.Response(
                    200,
                    json={
                        "access_token": "first-access",
                        "refresh_token": "first-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
            assert form["refresh_token"] == ["first-refresh"]
            return httpx.Response(
                200,
                json={
                    "access_token": "refreshed-access",
                    "refresh_token": "rotated-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(
        auth=oauth_provider(server, redirect_handler=redirect, callback_handler=callback),
        transport=transport,
    ) as client:
        assert (await client.get(server.url)).status_code == 200
    storage = FileTokenStorage(server)
    tokens = await storage.get_tokens()
    assert tokens is not None
    await storage.set_tokens(tokens.model_copy(update={"expires_in": 0}))
    async with httpx.AsyncClient(auth=oauth_provider(server), transport=transport) as client:
        assert (await client.get(server.url)).status_code == 200
    assert len(registrations) == 1 and len(grants) == 2
    tokens = await storage.get_tokens()
    assert tokens is not None and tokens.refresh_token == "rotated-refresh"


@pytest.mark.asyncio
async def test_legacy_server_refresh_uses_actual_exchange_endpoint_after_restart(
    tmp_path, monkeypatch
):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    server = MCPServerConfig(
        name="legacy", transport="streamable-http", url="https://resource.example/mcp", oauth=True
    )
    storage = FileTokenStorage(server)
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="legacy-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:8769/callback")],
            token_endpoint_auth_method="none",
        )
    )
    first = oauth_provider(server)
    await first._handle_token_response(
        httpx.Response(
            200,
            json={
                "access_token": "old",
                "refresh_token": "original-refresh",
                "token_type": "Bearer",
                "expires_in": 0,
            },
            request=httpx.Request("POST", "https://auth.example/custom-legacy-token"),
        )
    )
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            assert str(request.url) == "https://auth.example/custom-legacy-token"
            assert parse_qs(request.content.decode())["refresh_token"] == ["original-refresh"]
            return httpx.Response(
                200, json={"access_token": "fresh", "token_type": "Bearer", "expires_in": 3600}
            )
        assert request.headers["authorization"] == "Bearer fresh"
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        auth=oauth_provider(server), transport=httpx.MockTransport(respond)
    ) as client:
        assert (await client.get("https://resource.example/mcp")).status_code == 200
    assert len(requests) == 2
    saved = await storage.get_tokens()
    assert saved and saved.refresh_token == "original-refresh"
