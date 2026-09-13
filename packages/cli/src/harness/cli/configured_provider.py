"""Named OpenAI-compatible endpoints with explicit independent credentials."""

from __future__ import annotations

import os
from typing import Any

from harness.adapters.openai import OpenAIAdapter
from harness.core import Capabilities, ConfigurationError


class ConfiguredProvider(OpenAIAdapter):
    def __init__(self, name: str, *, settings: dict[str, Any], base_url: str | None = None):
        url = base_url or settings.get("base_url")
        if not isinstance(url, str) or not url:
            raise ConfigurationError(f"Provider {name} requires an explicit base_url")
        variable = settings.get("api_key_env")
        self.account = None
        if settings.get("oauth") is not None:
            from harness.cli.account_auth import AccountAuth, OAuthAccountConfig

            self.account = AccountAuth(
                name, OAuthAccountConfig.model_validate(settings["oauth"]), resource=url
            )
            variable = ""
        if not isinstance(variable, str):
            raise ConfigurationError(
                f"Provider {name} requires api_key_env; use an empty string for an unauthenticated local endpoint"
            )
        key = os.environ.get(variable) if variable else "local-endpoint"
        if not key:
            raise ConfigurationError(f"Provider {name} requires credential {variable}")
        super().__init__(api_key=key, base_url=url, timeout=float(settings.get("timeout", 120)))
        self.name = name
        self._capabilities = Capabilities(
            tool_use=settings.get("tool_use", True),
            input_media=settings.get("input_media", []),
            max_context_tokens=settings.get("max_context_tokens"),
        )

    async def capabilities(self) -> Capabilities:
        return self._capabilities.model_copy(deep=True)

    async def stream(self, *args: Any, **kwargs: Any):
        if self.account is not None:
            self.api_key = await self.account.access_token()
        async for event in super().stream(*args, **kwargs):
            yield event
