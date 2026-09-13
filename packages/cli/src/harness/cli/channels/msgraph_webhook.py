"""Basic Graph notifications with scoped durable ingress and local responses."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


def resource_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError("Graph requires a resource path")
    normalized = value.removeprefix("/")
    if (
        any(part in {"", ".", ".."} for part in normalized.split("/"))
        or any(char in normalized for char in ("\\", "%", "?", "#", ":"))
        or any(ord(char) < 32 for char in normalized)
    ):
        raise ValueError("Graph resource must be an unambiguous relative resource path")
    return normalized


class MSGraphWebhookTransport(WebhookTransport):
    name = "msgraph_webhook"
    limit = 4000
    # Subscription validation is a POST with text/plain, not a JSON notification.
    content_types = frozenset({"application/json", "text/plain"})

    async def authenticate(self) -> None:
        if (
            not 32 <= len(self.token.encode()) <= 128
            or not self.config.tenant_id
            or not self.config.subscription_id
            or not self.config.accepted_resources
        ):
            raise ChannelError(
                "Graph requires a 32-128 byte clientState token, tenant_id, "
                "subscription_id, and accepted_resources"
            )
        self.prefixes = tuple(resource_path(path) for path in self.config.accepted_resources)
        # Subscriptions expire and can be replaced. The stable account is the
        # tenant; each receipt, user identity and thread still binds subscription.
        self.identity = json.dumps(["msgraph", self.config.tenant_id])
        self.bot_id = self.config.tenant_id + ":" + self.config.subscription_id
        self.store: ChannelStore | None = None

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        try:
            await super().receive(store)
        finally:
            self.store = None

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    async def handle_request(
        self, request: web.Request, body: bytes, store: ChannelStore
    ) -> dict[str, Any] | web.Response:
        validation = request.query.getall("validationToken", [])
        if validation:
            if len(validation) != 1 or not validation[0] or len(validation[0]) > 4096:
                raise ValueError("Graph requires one bounded validation token")
            # This proves endpoint ownership only; it never authenticates work.
            return web.Response(text=validation[0], content_type="text/plain")
        if request.content_type != "application/json":
            raise ValueError("Graph notifications require JSON")
        await self.handle_event(request.headers, body, store)
        return web.Response(status=202)

    def _resource(self, raw: Any) -> str:
        resource = resource_path(raw)
        if not any(
            resource == prefix or resource.startswith(prefix + "/") for prefix in self.prefixes
        ):
            raise AuthenticationError("Graph resource is outside configured subscription scope")
        return resource

    def channel_id(self, thread_id: str) -> str:
        tenant, subscription, resource = json.loads(thread_id)
        if tenant != self.config.tenant_id or subscription != self.config.subscription_id:
            raise AuthenticationError("Graph conversation belongs to another subscription")
        return self._resource(resource)

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        payload = json.loads(body)
        if "validationTokens" in payload:
            raise ValueError(
                "Graph rich notifications are not supported; use includeResourceData=false"
            )
        values = payload["value"]
        if not isinstance(values, list) or not 1 <= len(values) <= 100:
            raise ValueError("Graph notification batch must contain 1-100 entries")
        pending = []
        for item in values:
            state = item.get("clientState", "")
            if (
                not self.token
                or not isinstance(state, str)
                or not hmac.compare_digest(state.encode(), self.token.encode())
                or item.get("tenantId") != self.config.tenant_id
                or item.get("subscriptionId") != self.config.subscription_id
            ):
                raise AuthenticationError("Graph notification authentication failed")
            if "encryptedContent" in item:
                raise ValueError("Encrypted rich notifications are not supported")
            lifecycle = item.get("lifecycleEvent")
            if lifecycle is not None:
                if lifecycle not in {"reauthorizationRequired", "subscriptionRemoved", "missed"}:
                    raise ValueError("Unknown Graph subscription lifecycle event")
                resource = "subscription"
                safe = {"lifecycleEvent": lifecycle}
            else:
                resource = self._resource(item["resource"])
                if item.get("changeType") not in {"created", "updated", "deleted"}:
                    raise ValueError("Unsupported Graph change type")
                data = item.get("resourceData", {})
                if not isinstance(data, dict):
                    raise ValueError("Graph resourceData must be metadata")
                safe = {
                    "resource": resource,
                    "changeType": item["changeType"],
                    "resourceData": {
                        key: data[key]
                        for key in ("id", "@odata.id", "@odata.type", "@odata.etag")
                        if key in data
                    },
                }
            # The clientState secret and arbitrary webhook fields never enter
            # transcripts, memory, or persisted notification payloads.
            safe.update(tenantId=self.config.tenant_id, subscriptionId=self.config.subscription_id)
            content = json.dumps(safe, sort_keys=True, separators=(",", ":"))
            receipt = hashlib.sha256(content.encode()).hexdigest()
            if item.get("changeType") == "updated" and not item.get("resourceData", {}).get(
                "@odata.etag"
            ):
                # Basic notifications can omit an event ID/version. Two updates
                # with identical metadata can represent different real edits;
                # permanent content hashing would silently lose later work.
                receipt = uuid.uuid4().hex
            if lifecycle is not None:
                pending.append((None, {"event": lifecycle, "id": receipt}))
                continue
            message = ChannelMessage(
                id=receipt,
                user_id=self.bot_id,
                channel_id=resource,
                thread_id=json.dumps(
                    [self.config.tenant_id, self.config.subscription_id, resource],
                    separators=(",", ":"),
                ),
                text="Microsoft Graph change notification (metadata, not resource contents):\n"
                + content,
            )
            pending.append((message, None))
        for message, lifecycle in pending:
            if lifecycle is not None:
                store.set("subscription_lifecycle", lifecycle)
                store.set("connection", "subscription_attention_required")
            else:
                self.accept(message, store)
        return {}

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self.channel_id(thread_id)
        if self.store is None:
            raise ChannelError("Graph local response store is not open")
        response = {"id": delivery_id, "thread_id": thread_id, "text": text, "destination": "local"}
        self.store.set("local_response:" + delivery_id, response)
        previous = self.store.get("local_responses", [])
        self.store.set(
            "local_responses",
            [response, *[item for item in previous if item["id"] != delivery_id]][:30],
        )

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        self.channel_id(thread_id)
        raise ChannelError("Graph notifications have no remote media reply destination")
