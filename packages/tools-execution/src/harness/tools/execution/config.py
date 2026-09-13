"""Explicit backend configuration; remote failures never select local execution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal[
        "local", "docker", "ssh", "singularity", "modal", "daytona", "vercel_sandbox"
    ] = "local"
    timeout_seconds: float = Field(default=120, gt=0, le=86400)
    max_output_bytes: int = Field(default=65536, ge=1024, le=16 * 1024 * 1024)
    max_file_bytes: int = Field(default=1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    max_processes: int = Field(default=32, ge=1, le=256)
    shell: str = "/bin/sh"
    windows_shell: str | None = None
    python_binary: str = "python3"
    docker_image: str | None = None
    docker_binary: str = "docker"
    docker_network: Literal["none", "bridge"] = "none"
    singularity_image: str | None = None
    singularity_binary: str = "singularity"
    ssh_host: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_user: str | None = None
    ssh_cwd: str | None = None
    ssh_binary: str = "ssh"
    cloud_cwd: str = "/tmp/harness-workspace"
    cloud_lifetime_seconds: int = Field(default=3600, ge=60, le=86400)
    modal_app: str = "harness-execution"
    modal_image: str = "python:3.12-slim"
    daytona_snapshot: str | None = None
    vercel_image: str | None = None
    import_paths: tuple[str, ...] = Field(default=(), max_length=256)
    export_paths: tuple[str, ...] = Field(default=(), max_length=256)
    export_directory: str = "artifacts/execution"
    transfer_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=256 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_backend(self) -> ExecutionConfig:
        if self.windows_shell is not None and (
            not self.windows_shell or "\x00" in self.windows_shell
        ):
            raise ValueError("windows_shell must name an explicit Bash executable")
        for value in (
            self.shell,
            self.python_binary,
            self.docker_binary,
            self.ssh_binary,
            self.singularity_binary,
        ):
            if not value or "\x00" in value:
                raise ValueError("execution binaries must be nonempty paths or executable names")
        if self.backend == "docker" and (
            not self.docker_image or self.docker_image.startswith("-")
        ):
            raise ValueError("docker backend requires an explicit docker_image")
        if self.backend == "singularity" and (
            not self.singularity_image or self.singularity_image.startswith("-")
        ):
            raise ValueError("singularity backend requires an explicit singularity_image")
        if self.backend == "ssh":
            if (
                not self.ssh_host
                or self.ssh_host.startswith("-")
                or any(char.isspace() or char == "\x00" for char in self.ssh_host)
            ):
                raise ValueError("ssh backend requires a valid explicit ssh_host")
            if self.ssh_user and (
                self.ssh_user.startswith("-")
                or any(char.isspace() or char in "@\x00" for char in self.ssh_user)
            ):
                raise ValueError("invalid ssh_user")
            if not self.ssh_cwd or not self.ssh_cwd.startswith("/") or "\x00" in self.ssh_cwd:
                raise ValueError("ssh backend requires an absolute ssh_cwd")
        if self.backend in {"modal", "daytona", "vercel_sandbox"}:
            if not self.cloud_cwd.startswith("/") or "\x00" in self.cloud_cwd:
                raise ValueError("cloud backend requires an absolute cloud_cwd")
            if self.timeout_seconds > self.cloud_lifetime_seconds:
                raise ValueError("command timeout may not exceed cloud sandbox lifetime")
        return self
