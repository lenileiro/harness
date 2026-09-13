import asyncio
import base64
import json

import pytest
from mcp import types
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl

from harness.core import ToolCall
from harness.tools.mcp import MCPServerConfig
from harness.tools.mcp.client import _result
from harness.tools.mcp.oauth import FileTokenStorage, authorization_callback, oauth_provider


async def test_private_tokens_expire_across_restart_and_profile_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "first"))
    config = MCPServerConfig(
        name="test", transport="streamable-http", url="https://mcp.example.test/mcp", oauth=True
    )
    store = FileTokenStorage(config)
    await store.set_tokens(
        OAuthToken(
            access_token="private-token",
            token_type="Bearer",
            refresh_token="private-refresh",
            expires_in=60,
        )
    )
    await store.set_client_info(
        OAuthClientInformationFull(
            client_id="client", redirect_uris=[AnyUrl("http://127.0.0.1:8766/callback")]
        )
    )
    loaded = await FileTokenStorage(config).get_tokens()
    assert (
        loaded
        and loaded.access_token == "private-token"
        and loaded.expires_in is not None
        and 0 <= loaded.expires_in <= 60
    )
    state = json.loads(store.path.read_text())
    state["expires_at"] = 1
    store.path.write_text(json.dumps(state))
    expired = await store.get_tokens()
    assert expired and expired.expires_in == 0
    assert store.path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "second"))
    assert await FileTokenStorage(config).get_tokens() is None


async def test_noninteractive_oauth_never_opens_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    config = MCPServerConfig(
        name="test", transport="streamable-http", url="https://mcp.example.test/mcp", oauth=True
    )
    provider = oauth_provider(config)
    assert provider.context.redirect_handler is not None
    with pytest.raises(RuntimeError, match="harness mcp login test"):
        await provider.context.redirect_handler("https://auth.example.test/")


async def test_callback_uses_loopback_and_returns_state(unused_tcp_port):
    async with authorization_callback(unused_tcp_port) as callback:
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        writer.write(
            b"GET /callback?code=code-value&state=state-value HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        )
        await writer.drain()
        assert b"200 OK" in await reader.read()
        assert await callback() == ("code-value", "state-value")
        writer.close()
        await writer.wait_closed()


def test_binary_mcp_content_becomes_portable_media():
    data = base64.b64encode(b"binary").decode()
    response = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="screen"),
            types.ImageContent(type="image", data=data, mimeType="image/png"),
            types.AudioContent(type="audio", data=data, mimeType="audio/wav"),
        ]
    )
    result = _result(ToolCall(id="m", name="mcp__screen", arguments={}), response, "test", 10000)
    assert [item.kind for item in result.attachments] == ["image", "audio"]
    assert "unavailable" not in result.content
    assert result.metadata is not None and "unsupported_content" not in result.metadata
