"""Official MCP OAuth/PKCE flow with profile-scoped private token persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from time import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
from filelock import FileLock
from pydantic import AnyUrl

from harness.core.paths import read_regular_file, user_home
from harness.tools.mcp.config import MCPServerConfig
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)


class PersistentOAuthProvider(OAuthClientProvider):
    """Restore expiry and discovered issuer endpoints across SDK client lifetimes.

    The SDK's default initializer restores tokens but not expiry or discovery.
    Persist both together so a restarted process refreshes at the same verified
    authorization server, and serialize rotation across clients/processes.
    """

    async def _initialize(self) -> None:
        await super()._initialize()
        storage = self.context.storage
        assert isinstance(storage, FileTokenStorage)
        state = await asyncio.to_thread(storage._load)
        metadata = state.get("oauth_metadata")
        self._bound_token_endpoint = state.get("verified_token_endpoint")
        if (
            self.context.current_tokens is not None
            and metadata is None
            and not self._bound_token_endpoint
        ):
            # Older credentials lack a bound refresh endpoint. Reauthorize;
            # never send their refresh token to a guessed resource-server URL.
            self.context.clear_tokens()
            return
        if metadata is not None:
            self.context.oauth_metadata = OAuthMetadata.model_validate(metadata)
        self.context.auth_server_url = state.get("auth_server_url")
        resource = state.get("protected_resource_metadata")
        if resource is not None:
            self.context.protected_resource_metadata = ProtectedResourceMetadata.model_validate(
                resource
            )
        self.context.token_expiry_time = state.get("expires_at")

    async def _refresh_token(self) -> httpx.Request:
        request = await super()._refresh_token()
        endpoint = getattr(self, "_bound_token_endpoint", None)
        if self.context.oauth_metadata is None and isinstance(endpoint, str) and endpoint:
            # Legacy servers may omit RFC8414 metadata. The actual successful
            # exchange endpoint, never a guess, is the bound refresh destination.
            request.url = httpx.URL(endpoint)
        return request

    async def _handle_token_response(self, response: httpx.Response) -> None:
        if response.status_code != 200:
            raise RuntimeError(f"MCP OAuth token exchange failed (HTTP {response.status_code})")
        await super()._handle_token_response(response)
        storage = self.context.storage
        assert isinstance(storage, FileTokenStorage)
        await asyncio.to_thread(
            storage._update,
            verified_token_endpoint=str(response.request.url),
            oauth_metadata=self.context.oauth_metadata.model_dump(mode="json")
            if self.context.oauth_metadata
            else None,
            auth_server_url=self.context.auth_server_url,
            protected_resource_metadata=self.context.protected_resource_metadata.model_dump(
                mode="json"
            )
            if self.context.protected_resource_metadata
            else None,
        )

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        if response.status_code in {429, 500, 502, 503, 504}:
            raise RuntimeError(
                "MCP OAuth refresh temporarily unavailable; saved credentials retained"
            )
        previous = self.context.current_tokens
        success = await super()._handle_refresh_response(response)
        storage = self.context.storage
        assert isinstance(storage, FileTokenStorage)
        if not success and response.status_code not in {429, 500, 502, 503, 504}:
            await asyncio.to_thread(storage._update, tokens=None, expires_at=None)
        if (
            success
            and self.context.current_tokens
            and not self.context.current_tokens.refresh_token
            and previous
        ):
            self.context.current_tokens.refresh_token = previous.refresh_token
            await storage.set_tokens(self.context.current_tokens)
        return success

    async def _auth_flow(self, request: httpx.Request):
        storage = self.context.storage
        assert isinstance(storage, FileTokenStorage)
        storage.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = FileLock(
            storage.path.with_suffix(".flow.lock"), timeout=0, mode=0o600, thread_local=False
        )
        from contextlib import aclosing

        from filelock import Timeout

        deadline = asyncio.get_running_loop().time() + 300
        while True:
            try:
                lock.acquire()
                break
            except Timeout:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("Another MCP authorization is still running") from None
                await asyncio.sleep(0.05)
        try:
            self._initialized = False
            async with aclosing(super()._auth_flow(request)) as flow:
                outgoing = await anext(flow)
                while True:
                    response = yield outgoing
                    try:
                        outgoing = await flow.asend(response)
                    except StopAsyncIteration:
                        return
        finally:
            lock.release()


class FileTokenStorage:
    def __init__(self, server: MCPServerConfig):
        identity = hashlib.sha256(
            json.dumps(
                [server.name, server.url, server.oauth_redirect_port, server.oauth_scopes]
            ).encode()
        ).hexdigest()
        self.path = user_home() / "auth/mcp" / f"{identity}.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(read_regular_file(self.path, max_bytes=1024 * 1024))

    def _update(self, **changes: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.path.with_suffix(".lock"), mode=0o600):
            state = {**self._load(), **changes}
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}")
            try:
                fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w") as stream:
                    json.dump(state, stream)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)

    async def get_tokens(self) -> OAuthToken | None:
        state = await asyncio.to_thread(self._load)
        if not state.get("tokens"):
            return None
        tokens = dict(state["tokens"])
        if state.get("expires_at") is not None:
            tokens["expires_in"] = max(0, int(state["expires_at"] - time()))
        return OAuthToken.model_validate(tokens)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = time() + tokens.expires_in if tokens.expires_in is not None else None
        await asyncio.to_thread(
            self._update, tokens=tokens.model_dump(mode="json"), expires_at=expires_at
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        value = (await asyncio.to_thread(self._load)).get("client")
        return OAuthClientInformationFull.model_validate(value) if value else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(self._update, client=client_info.model_dump(mode="json"))


def oauth_provider(
    server: MCPServerConfig, *, redirect_handler=None, callback_handler=None
) -> OAuthClientProvider:
    async def login_required(url: str) -> None:
        raise RuntimeError(f"MCP OAuth login required: harness mcp login {server.name}")

    assert server.url is not None
    return PersistentOAuthProvider(
        server_url=server.url,
        client_metadata=OAuthClientMetadata(
            client_name="Harness",
            redirect_uris=[AnyUrl(f"http://127.0.0.1:{server.oauth_redirect_port}/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            scope=server.oauth_scopes,
        ),
        storage=FileTokenStorage(server),
        redirect_handler=redirect_handler or login_required,
        callback_handler=callback_handler,
        timeout=300,
    )


@asynccontextmanager
async def authorization_callback(port: int):
    """Bounded loopback callback. State/PKCE verification belongs to the SDK."""
    queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue(maxsize=1)
    connections: set[asyncio.Task] = set()

    async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        try:
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            line = header.split(b"\r\n", 1)[0].decode("ascii")
            method, target, _ = line.split(" ", 2)
            parsed = urlparse(target)
            params = parse_qs(parsed.query)
            valid = (
                method == "GET"
                and parsed.path == "/callback"
                and bool(params.get("code"))
                and bool(params.get("state"))
            )
            if valid and not queue.full():
                queue.put_nowait((params["code"][0], params["state"][0]))
            body = (
                b"Authorization received. Return to Harness."
                if valid
                else b"Invalid authorization callback."
            )
            writer.write(
                b"HTTP/1.1 "
                + (b"200 OK" if valid else b"400 Bad Request")
                + b"\r\nContent-Type: text/plain\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        except (
            TimeoutError,
            ValueError,
            OSError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            finally:
                connections.discard(task)

    listener = await asyncio.start_server(receive, "127.0.0.1", port, limit=16384)

    async def callback():
        return await asyncio.wait_for(queue.get(), timeout=300)

    try:
        yield callback
    finally:
        listener.close()
        await listener.wait_closed()
        active = tuple(connections)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
