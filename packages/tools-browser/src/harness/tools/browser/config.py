from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class BrowserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["local", "cdp"] = "local"
    executable_path: str | None = None
    cdp_url: SecretStr | None = None
    headless: bool = True
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    viewport_width: int = Field(default=1280, ge=320, le=3840)
    viewport_height: int = Field(default=800, ge=240, le=2160)
    max_snapshot_chars: int = Field(default=20000, ge=1000, le=100000)
    max_tabs: int = Field(default=10, ge=1, le=50)
    max_downloads: int = Field(default=32, ge=1, le=256)
    max_artifact_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    artifact_directory: str = "artifacts/browser"

    @model_validator(mode="after")
    def validate_backend(self) -> BrowserConfig:
        if self.backend == "cdp":
            if self.cdp_url is None:
                raise ValueError("CDP backend requires an explicit cdp_url")
            url = urlsplit(self.cdp_url.get_secret_value())
            if url.scheme not in {"http", "https", "ws", "wss"} or not url.hostname:
                raise ValueError("cdp_url must be an HTTP(S) or WS(S) endpoint")
        return self
