"""Vision attachments and bounded image/transcription/speech HTTP workflows."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.core import (
    ApprovalDecision,
    ConfigurationError,
    MediaAttachment,
    Tool,
    ToolCall,
    ToolResult,
)
from harness.core.media_metadata import audio_duration_ms


class MediaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    image_model: str | None = None
    transcription_model: str | None = None
    speech_model: str | None = None
    voice: str = "alloy"
    timeout: float = Field(default=120, gt=0, le=900)
    max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    artifact_directory: str = "artifacts/media"
    image_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_url(self):
        parsed = httpx.URL(self.base_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.host in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Media API must use HTTPS (HTTP allowed for loopback fixtures)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Media API URL must not contain credentials, query, or fragment")
        if self.image_parameters.get("output_format", "png") not in {"png", "jpeg", "webp"}:
            raise ValueError("Image output_format must be png, jpeg, or webp")
        return self


def _workspace_path(cwd: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if ".harness" in {part.casefold() for part in candidate.relative_to(cwd).parts}:
        raise ValueError("Harness private state cannot be used as a media artifact")
    resolved = candidate.resolve()
    relative = resolved.relative_to(cwd)
    if ".harness" in {part.casefold() for part in relative.parts}:
        raise ValueError("Harness private state cannot be used as a media artifact")
    return resolved


class MediaToolset:
    def __init__(
        self,
        config: MediaConfig,
        *,
        cwd: Path,
        transport: httpx.AsyncBaseTransport | None = None,
        token_provider: Callable[[], Awaitable[str]] | None = None,
    ):
        self.config = config
        self.cwd = cwd.resolve()
        self._transport = transport
        self._token_provider = token_provider
        self.client: httpx.AsyncClient | None = None
        self.tools: list[Tool] = []

    async def __aenter__(self):
        if self.config.enabled:
            self.tools.append(_MediaTool(self, "read_media"))
            configured = [
                self.config.image_model,
                self.config.transcription_model,
                self.config.speech_model,
            ]
            if any(configured):
                key = (
                    "account-managed"
                    if self._token_provider
                    else os.environ.get(self.config.api_key_env, "")
                )
                if not key:
                    raise ConfigurationError(f"Missing media credential {self.config.api_key_env}")
                self.client = httpx.AsyncClient(
                    base_url=self.config.base_url.rstrip("/") + "/",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=self.config.timeout,
                    follow_redirects=False,
                    trust_env=False,
                    transport=self._transport,
                )
            for model, name in zip(
                configured, ["image_generate", "audio_transcribe", "speech_generate"], strict=True
            ):
                if model:
                    self.tools.append(_MediaTool(self, name))
        return self

    async def __aexit__(self, *_):
        if self.client:
            await self.client.aclose()
        self.client = None
        self.tools = []

    async def request(self, endpoint: str, **kwargs: Any) -> tuple[bytes, str]:
        if self.client is None:
            raise ConfigurationError("Media toolset is closed or has no API credential")
        if self._token_provider is not None:
            kwargs["headers"] = {"Authorization": f"Bearer {await self._token_provider()}"}
        async with self.client.stream("POST", endpoint, **kwargs) as response:
            if not response.is_success:
                raise ValueError(
                    f"Media provider returned HTTP {response.status_code}; request was not retried"
                )
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > self.config.max_bytes * 4 // 3 + 65536:
                    raise ValueError("Media response exceeds configured size limit")
            return bytes(data), response.headers.get("content-type", "")

    def persist(self, media: MediaAttachment, suffix: str) -> Path:
        root = _workspace_path(self.cwd, self.config.artifact_directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{uuid4().hex}{suffix}"
        raw = base64.b64decode(media.data or "", validate=True)
        if len(raw) > self.config.max_bytes:
            raise ValueError("Media exceeds configured size limit")
        with path.open("xb") as output:
            output.write(raw)
        return path


class _MediaTool:
    def __init__(self, owner: MediaToolset, name: str):
        self.owner = owner
        self.name = name
        self.approval: ApprovalDecision = "auto" if name == "read_media" else "prompt"
        self.effect_scope: Literal["read_only", "external_side_effect"] = (
            "read_only" if name == "read_media" else "external_side_effect"
        )
        self.description = {
            "read_media": "Read a workspace image, audio, or document as native model input.",
            "image_generate": "Generate an image through the configured media API and save it as an artifact.",
            "audio_transcribe": "Upload a workspace audio file to the configured transcription API.",
            "speech_generate": "Generate speech through the configured API and save a playable audio artifact.",
        }[name]
        parameter = (
            "path"
            if name in {"read_media", "audio_transcribe"}
            else "text"
            if name == "speech_generate"
            else "prompt"
        )
        self.parameters_schema = {
            "type": "object",
            "properties": {parameter: {"type": "string", "minLength": 1, "maxLength": 16000}},
            "required": [parameter],
            "additionalProperties": False,
        }
        self._parameter = parameter

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            value = call.arguments.get(self._parameter)
            if not isinstance(value, str) or not value.strip() or len(value) > 16000:
                raise ValueError(f"{self._parameter} must be nonempty and at most 16000 characters")
            config = self.owner.config
            if self.name in {"read_media", "audio_transcribe"}:
                media = await asyncio.to_thread(
                    lambda: MediaAttachment.from_file(_workspace_path(self.owner.cwd, value))
                )
                if len(base64.b64decode(media.data or "")) > config.max_bytes:
                    raise ValueError("Input media exceeds configured size limit")
                if self.name == "read_media":
                    return ToolResult(
                        tool_call_id=call.id,
                        name=self.name,
                        content=f"Media: {media.name}",
                        attachments=[media],
                    )
                if media.kind != "audio":
                    raise ValueError("Transcription requires an audio file")
                raw, _ = await self.owner.request(
                    "audio/transcriptions",
                    files={
                        "file": (media.name, base64.b64decode(media.data or ""), media.mime_type)
                    },
                    data={"model": config.transcription_model},
                )
                text = json.loads(raw).get("text")
                if not isinstance(text, str):
                    raise ValueError("Provider returned no transcription text")
                return ToolResult(tool_call_id=call.id, name=self.name, content=text)
            if self.name == "image_generate":
                raw, _ = await self.owner.request(
                    "images/generations",
                    json={
                        **config.image_parameters,
                        "model": config.image_model,
                        "prompt": value,
                        "n": 1,
                    },
                )
                data = json.loads(raw).get("data", [])
                if (
                    not isinstance(data, list)
                    or not data
                    or not isinstance(data[0], dict)
                    or not isinstance(data[0].get("b64_json"), str)
                ):
                    raise ValueError("Image provider must return b64_json for a durable artifact")
                output_format = config.image_parameters.get("output_format", "png")
                media = MediaAttachment(
                    kind="image", mime_type=f"image/{output_format}", data=data[0]["b64_json"]
                )
                suffix = f".{output_format}"
            else:
                raw, _ = await self.owner.request(
                    "audio/speech",
                    json={
                        "model": config.speech_model,
                        "voice": config.voice,
                        "input": value,
                        "response_format": "mp3",
                    },
                )
                media = MediaAttachment(
                    kind="audio",
                    mime_type="audio/mpeg",
                    data=base64.b64encode(raw).decode(),
                    model_visible=False,
                    duration_ms=await asyncio.to_thread(audio_duration_ms, raw),
                )
                suffix = ".mp3"
            path = await asyncio.to_thread(self.owner.persist, media, suffix)
            media.name = path.name
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"Saved {path.relative_to(self.owner.cwd)}",
                attachments=[media],
                metadata={"path": str(path), "generated": True},
            )
        except (OSError, ValueError, httpx.HTTPError, KeyError, TypeError) as exc:
            # HTTP exceptions can contain credential-bearing request URLs.
            message = (
                "Media transport failed; request was not retried"
                if isinstance(exc, httpx.HTTPError)
                else str(exc)
            )
            return ToolResult(tool_call_id=call.id, name=self.name, content=message, is_error=True)


__all__ = ["MediaConfig", "MediaToolset"]
