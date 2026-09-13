"""Home Assistant REST capability with fixed endpoint and explicit grants."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from harness.core.schemas import ToolCall, ToolResult

_ENTITY = re.compile(r"[a-z][a-z0-9_]*\.[a-z0-9][a-z0-9_]*\Z")
_SERVICE = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\Z")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TARGET_FIELDS = {"entity_id", "device_id", "area_id", "floor_id", "label_id", "target"}


class HomeAssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    enabled: bool = False
    base_url: str | None = None
    token_env: str = "HOME_ASSISTANT_TOKEN"
    entities: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    service_fields: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    timeout_seconds: FiniteFloat = Field(default=20, ge=1, le=120)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1024, le=4 * 1024 * 1024)

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.enabled and not self.base_url:
            raise ValueError("enabled Home Assistant requires base_url")
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("base_url must be HTTP(S) without credentials, query or fragment")
        if not _VARIABLE.fullmatch(self.token_env):
            raise ValueError("token_env must name one environment variable")
        if (
            len(self.entities) > 1000
            or len(set(self.entities)) != len(self.entities)
            or any(not _ENTITY.fullmatch(value) for value in self.entities)
        ):
            raise ValueError("entities must contain unique exact entity IDs (at most 1000)")
        if (
            len(self.services) > 1000
            or len(set(self.services)) != len(self.services)
            or any(not _SERVICE.fullmatch(value) for value in self.services)
        ):
            raise ValueError(
                "services must contain unique exact domain.service names (at most 1000)"
            )
        for service, fields in self.service_fields.items():
            if service not in self.services or any(
                name in _TARGET_FIELDS or not _VARIABLE.fullmatch(name) for name in fields
            ):
                raise ValueError(
                    "service_fields must grant non-target fields for an allowed service"
                )
        return self


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    entity_id: str = Field(min_length=1, max_length=256)


class ServiceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    service: str = Field(min_length=1, max_length=128)
    entity_ids: list[str] = Field(min_length=1, max_length=64)
    data: dict[str, Any] = Field(default_factory=dict)


class HomeAssistantError(ValueError):
    pass


class HomeAssistantToolset:
    def __init__(
        self,
        config: HomeAssistantConfig,
        *,
        environment: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self._environment = environment
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.tools: list[HomeAssistantTool] = []

    async def __aenter__(self) -> Self:
        if self._client is not None:
            raise HomeAssistantError("Home Assistant toolset is already open")
        if not self.config.enabled:
            return self
        environment = self._environment if self._environment is not None else os.environ
        token = environment.get(self.config.token_env)
        if not token or not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token):
            raise HomeAssistantError(
                "Home Assistant token environment reference is missing or invalid"
            )
        base = (self.config.base_url or "").rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        self._client = httpx.AsyncClient(
            base_url=base + "/api/",
            headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
            timeout=self.config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        )
        names = [
            "homeassistant_list_entities",
            "homeassistant_get_state",
            "homeassistant_list_services",
        ]
        if self.config.services and self.config.entities:
            names.append("homeassistant_call_service")
        self.tools = [HomeAssistantTool(self, name) for name in names]
        return self

    async def __aexit__(self, *args) -> None:
        client, self._client = self._client, None
        self.tools = []
        if client is not None:
            await client.aclose()

    async def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise HomeAssistantError("Home Assistant toolset is closed")
        try:
            async with self._client.stream(method, path, json=data) as response:
                if 300 <= response.status_code < 400:
                    raise HomeAssistantError(
                        "Home Assistant redirects are not followed; configure the final trusted endpoint"
                    )
                if response.status_code >= 400:
                    detail = (
                        "Home Assistant rejected the request"
                        if response.status_code < 500
                        else "Home Assistant server error; inspect device state before retrying a service action"
                    )
                    raise HomeAssistantError(f"{detail} (HTTP {response.status_code})")
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > self.config.max_response_bytes:
                        raise HomeAssistantError(
                            "Home Assistant response exceeded its configured size limit"
                        )
            return json.loads(raw)
        except httpx.HTTPError:
            raise HomeAssistantError(
                "Home Assistant request failed; a service action may have reached the server, so inspect device state before retrying"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HomeAssistantError(
                "Home Assistant returned invalid JSON; inspect state before retrying a service action"
            ) from None


class HomeAssistantTool:
    def __init__(self, toolset: HomeAssistantToolset, name: str):
        self.toolset = toolset
        self.name = name
        mutation = name == "homeassistant_call_service"
        self.approval: Literal["auto", "prompt", "deny"] = "prompt" if mutation else "auto"
        self.effect_scope = "external_side_effect" if mutation else "read_only"
        self.phases = ("*",)
        self.description = {
            "homeassistant_list_entities": "List only explicitly configured Home Assistant entity IDs. No network request.",
            "homeassistant_get_state": "Read one explicitly allowed Home Assistant entity's current state.",
            "homeassistant_list_services": "Inspect only explicitly allowed Home Assistant services and their granted data fields.",
            "homeassistant_call_service": "Call an explicitly allowed Home Assistant service for exact allowed entities, through approval. Only configured data fields are accepted. No automatic retry.",
        }[name]
        self.parameters_schema = (
            ServiceArguments
            if mutation
            else StateArguments
            if name == "homeassistant_get_state"
            else EmptyArguments
        ).model_json_schema()
        if mutation:
            self.parameters_schema["properties"]["service"]["enum"] = list(toolset.config.services)
            self.parameters_schema["properties"]["entity_ids"]["items"]["enum"] = list(
                toolset.config.entities
            )
        elif name == "homeassistant_get_state":
            self.parameters_schema["properties"]["entity_id"]["enum"] = list(
                toolset.config.entities
            )

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            result = await self.execute(call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=json.dumps(result, ensure_ascii=False, allow_nan=False),
                metadata={"provider": "homeassistant"},
            )
        except HomeAssistantError as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)
        except (ValueError, TypeError, KeyError, RecursionError):
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="Invalid Home Assistant arguments or response shape; inspect the configured grants",
                is_error=True,
            )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        config = self.toolset.config
        if self.name == "homeassistant_get_state":
            request = StateArguments.model_validate(arguments)
            if request.entity_id not in config.entities:
                raise HomeAssistantError(
                    "Entity is not allowed by this Home Assistant configuration"
                )
            state = await self.toolset.request("GET", "states/" + request.entity_id)
            if not isinstance(state, dict) or state.get("entity_id") != request.entity_id:
                raise HomeAssistantError("Home Assistant returned a mismatched entity")
            return state
        if self.name == "homeassistant_call_service":
            request = ServiceArguments.model_validate(arguments)
            if request.service not in config.services or any(
                entity not in config.entities for entity in request.entity_ids
            ):
                raise HomeAssistantError(
                    "Service or entity is not allowed by this Home Assistant configuration"
                )
            if len(set(request.entity_ids)) != len(request.entity_ids):
                raise HomeAssistantError("Entity IDs must be unique")
            if set(request.data) - set(config.service_fields.get(request.service, ())):
                raise HomeAssistantError(
                    "Service data contains fields not explicitly granted by the operator"
                )
            payload = {**request.data, "entity_id": request.entity_ids}
            if len(json.dumps(payload, allow_nan=False).encode()) > 64000:
                raise HomeAssistantError("Service data exceeds 64,000 bytes")
            domain, service = request.service.split(".", 1)
            result = await self.toolset.request("POST", f"services/{domain}/{service}", payload)
            if not isinstance(result, list):
                raise HomeAssistantError(
                    "Service response is not a state list; inspect state before retrying"
                )
            return {
                "service": request.service,
                "entity_ids": request.entity_ids,
                "states": [
                    item
                    for item in result
                    if isinstance(item, dict) and item.get("entity_id") in config.entities
                ],
            }
        EmptyArguments.model_validate(arguments)
        if self.name == "homeassistant_list_entities":
            return {"entities": list(config.entities)}
        response = await self.toolset.request("GET", "services")
        if not isinstance(response, list):
            raise HomeAssistantError("Home Assistant returned an invalid service catalog")
        services = []
        for domain in response:
            if not isinstance(domain, dict) or not isinstance(domain.get("services"), dict):
                raise HomeAssistantError("Home Assistant returned an invalid service catalog")
            for service, details in domain["services"].items():
                name = f"{domain.get('domain')}.{service}"
                if name in config.services and isinstance(details, dict):
                    fields = details.get("fields", {})
                    if not isinstance(fields, dict):
                        raise HomeAssistantError("Home Assistant returned invalid service fields")
                    services.append(
                        {
                            "service": name,
                            "description": details.get("description", ""),
                            "fields": {
                                key: value
                                for key, value in fields.items()
                                if key in config.service_fields.get(name, ())
                            },
                            "entities": list(config.entities),
                        }
                    )
        return {"services": services}
