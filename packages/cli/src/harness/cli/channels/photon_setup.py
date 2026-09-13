"""Photon's public device login and project/user bootstrap, without secret rotation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from harness.cli.channels.transports import ChannelError

HOST = "https://app.photon.codes"


class PhotonSetup:
    def __init__(self, client: httpx.AsyncClient, *, sleep=asyncio.sleep):
        self.client, self.sleep = client, sleep
        self.token = ""

    async def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        response = await self.client.request(
            method,
            HOST + path,
            json=data,
            headers={"Authorization": "Bearer " + self.token} if self.token else {},
        )
        if not response.is_success or len(response.content) > 2**20:
            raise ChannelError(f"Photon setup request rejected (HTTP {response.status_code})")
        value = response.json()
        if isinstance(value, dict) and (value.get("error") or value.get("succeed") is False):
            raise ChannelError("Photon setup operation was rejected")
        return value

    async def login(
        self, display: Callable[[str, str], Awaitable[None]], *, timeout_seconds: int = 600
    ) -> None:
        code = await self.request(
            "POST",
            "/api/auth/device/code",
            {"client_id": "photon-cli", "scope": "openid profile email"},
        )
        uri, user_code = code.get("verification_uri"), code.get("user_code")
        parsed = urlsplit(uri or "")
        if (
            parsed.scheme != "https"
            or parsed.hostname != "app.photon.codes"
            or parsed.username
            or parsed.password
            or parsed.fragment
            or not isinstance(user_code, str)
            or not user_code
            or len(user_code) > 100
        ):
            raise ChannelError("Photon returned an invalid device authorization destination")
        await display(uri, user_code)
        delay = max(1, min(60, int(code.get("interval") or 5)))
        async with asyncio.timeout(
            min(timeout_seconds, max(1, int(code.get("expires_in") or 600)))
        ):
            while True:
                await self.sleep(delay)
                response = await self.client.post(
                    HOST + "/api/auth/device/token",
                    json={
                        "client_id": "photon-cli",
                        "device_code": code["device_code"],
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                if response.status_code == 429:
                    delay = min(60, delay + 10)
                    continue
                if len(response.content) > 16 * 1024:
                    raise ChannelError("Photon device response exceeds its size limit")
                body = response.json()
                if response.status_code == 400 and body.get("error") in {
                    "authorization_pending",
                    "slow_down",
                }:
                    if body["error"] == "slow_down":
                        delay = min(60, delay + 5)
                    continue
                if not response.is_success:
                    raise ChannelError("Photon device authorization was denied or expired")
                nested = body.get("data", {})
                candidates = [
                    body.get("access_token"),
                    body.get("accessToken"),
                    nested.get("access_token") if isinstance(nested, dict) else None,
                    response.headers.get("set-auth-token"),
                ]
                for candidate in candidates:
                    if not isinstance(candidate, str) or not candidate.strip():
                        continue
                    self.token = candidate.removeprefix("Bearer ").strip()
                    try:
                        session = await self.request("GET", "/api/auth/get-session")
                        if not isinstance(session, dict) or not session.get("user"):
                            continue
                        await self.request("GET", "/api/projects/")
                    except ChannelError:
                        continue
                    return
                self.token = ""
                raise ChannelError("Photon did not issue a project-authorized session")

    async def setup(
        self, *, phone: str, project_id: str = "", create_project: str = ""
    ) -> dict[str, str]:
        if not re.fullmatch(r"\+[1-9][0-9]{6,14}", phone):
            raise ValueError("Photon phone must use E.164 format")
        if bool(project_id) == bool(create_project):
            raise ValueError("Choose exactly one of --project or --create-project")
        if not self.token:
            raise ChannelError("Photon device login is required")
        if create_project:
            if len(create_project) > 200:
                raise ValueError("Photon project name is too long")
            projects = await self.request("GET", "/api/projects")
            if not isinstance(projects, list):
                raise ChannelError("Photon project list has an unsupported shape")
            matches = [item for item in projects if item.get("name") == create_project]
            if len(matches) > 1:
                raise ChannelError("Photon project name is ambiguous; choose --project ID")
            project = (
                matches[0]
                if matches
                else await self.request(
                    "POST",
                    "/api/projects",
                    {
                        "name": create_project,
                        "location": "United States",
                        "template": False,
                        "observability": False,
                        "platforms": ["imessage"],
                    },
                )
            )
            project_id = project.get("id", "")
        if not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", project_id):
            raise ChannelError("Photon project ID is invalid")
        path = "/api/projects/" + quote(project_id, safe="")
        project = await self.request("GET", path)
        secret = project.get("projectSecret")
        if project.get("id") != project_id or not isinstance(secret, str) or not secret:
            raise ChannelError(
                "Photon project has no readable secret; configure one in its dashboard before setup"
            )
        users = await self.request("GET", path + "/spectrum/users")
        if not isinstance(users, dict) or not isinstance(users.get("users"), list):
            raise ChannelError("Photon user list has an unsupported shape")
        matches = [item for item in users["users"] if item.get("phoneNumber") == phone]
        if len(matches) > 1:
            raise ChannelError("Photon phone maps to multiple users; resolve the project accounts")
        user = (
            matches[0]
            if matches
            else (
                await self.request(
                    "POST", path + "/spectrum/users", {"phoneNumber": phone, "sendInvite": False}
                )
            ).get("user", {})
        )
        if (
            not isinstance(user.get("id"), str)
            or not user["id"]
            or user.get("phoneNumber") != phone
        ):
            raise ChannelError("Photon did not register the requested phone identity")
        return {
            "project_id": project_id,
            "project_secret": secret,
            "user_id": user["id"],
            "phone": phone,
            "assigned_phone": str(user.get("assignedPhoneNumber") or ""),
        }
