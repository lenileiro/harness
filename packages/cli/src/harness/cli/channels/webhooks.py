"""Bounded HTTP event ingestion. Authentication precedes durable acknowledgment."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelStore

MAX_EVENT_BYTES = 1024 * 1024


class AuthenticationError(ChannelError):
    pass


class WebhookTransport(Transport):
    content_types = frozenset({"application/json"})
    http_methods = ("POST",)

    async def handle_request(
        self, request: web.Request, body: bytes, store: ChannelStore
    ) -> dict[str, Any] | web.Response:
        return await self.handle_event(request.headers, body, store)

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        raise NotImplementedError

    def application(self, store: ChannelStore) -> web.Application:
        path = self.config.webhook_path
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ChannelError("webhook_path must be an absolute HTTP path")
        capacity = asyncio.Semaphore(32)

        async def receive(request: web.Request) -> web.Response:
            if capacity.locked():
                return web.json_response({"error": "busy"}, status=503)
            async with capacity:
                try:
                    if request.method != "GET" and request.content_type not in self.content_types:
                        return web.json_response({"error": "unsupported content type"}, status=415)
                    if request.headers.get("Content-Encoding", "identity") != "identity":
                        return web.json_response(
                            {"error": "compressed events unsupported"}, status=415
                        )
                    # Reject duplicate authority headers instead of guessing which
                    # value a proxy, framework, or signature verifier selected.
                    names = [key.decode("ascii").lower() for key, _ in request.raw_headers]
                    if len(names) != len(set(names)):
                        return web.json_response({"error": "duplicate headers"}, status=400)
                    body = await asyncio.wait_for(request.read(), timeout=10)
                    result = await asyncio.wait_for(
                        self.handle_request(request, body, store), timeout=15
                    )
                    return result if isinstance(result, web.Response) else web.json_response(result)
                except AuthenticationError:
                    return web.json_response({"error": "unauthorized"}, status=401)
                except web.HTTPRequestEntityTooLarge:
                    return web.json_response({"error": "event too large"}, status=413)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    return web.json_response({"error": "invalid event"}, status=400)
                except TimeoutError:
                    return web.json_response({"error": "timeout"}, status=503)
                except Exception:
                    return web.json_response({"error": "temporarily unavailable"}, status=503)

        app = web.Application(client_max_size=MAX_EVENT_BYTES)
        for method in self.http_methods:
            app.router.add_route(method, path, receive)
        return app

    async def receive(self, store: ChannelStore) -> None:
        runner = web.AppRunner(self.application(store), access_log=None, shutdown_timeout=5)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self.config.listen_host, self.config.listen_port)
            await site.start()
            store.set("connection", "listening")
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
