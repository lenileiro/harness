import json
import os

import httpx
import pytest
import typer
from typer.testing import CliRunner

from harness.cli import channel_photon_commands
from harness.cli.channel_photon_commands import photon_account_path, save_photon_account
from harness.cli.channels.photon_setup import HOST, PhotonSetup
from harness.cli.channels.runtime import build_transport
from harness.cli.channels.transports import ChannelError
from harness.core.gateway_channels import ChannelConfig


async def test_photon_device_slow_down_project_create_reuse_secret_and_phone_no_invite():
    requests, displayed, sleeps = [], [], []
    projects, users = [], []
    polls = 0

    def http(request):
        nonlocal polls
        assert str(request.url).startswith(HOST + "/api/")
        path = request.url.path
        body = json.loads(request.content or b"{}")
        requests.append((request.method, path, body))
        if path == "/api/auth/device/code":
            assert body["client_id"] == "photon-cli"
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": HOST + "/device",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        if path == "/api/auth/device/token":
            assert body["device_code"] == "device-secret"
            polls += 1
            if polls == 1:
                return httpx.Response(400, json={"error": "slow_down"})
            return httpx.Response(200, json={"access_token": "dashboard-secret"})
        assert request.headers["Authorization"] == "Bearer dashboard-secret"
        if path == "/api/auth/get-session":
            return httpx.Response(200, json={"user": {"id": "operator"}})
        if path.rstrip("/") == "/api/projects":
            if request.method == "POST":
                assert body["platforms"] == ["imessage"]
                projects.append({"id": "project", "name": body["name"]})
                return httpx.Response(200, json=projects[0])
            return httpx.Response(200, json=projects)
        if path == "/api/projects/project":
            return httpx.Response(200, json={"id": "project", "projectSecret": "project-secret"})
        assert path == "/api/projects/project/spectrum/users"
        if request.method == "POST":
            assert body == {"phoneNumber": "+15551234567", "sendInvite": False}
            users.append(
                {
                    "id": "user",
                    "phoneNumber": body["phoneNumber"],
                    "assignedPhoneNumber": "+15557654321",
                }
            )
            return httpx.Response(200, json={"user": users[0]})
        return httpx.Response(200, json={"users": users})

    async def display(uri, code):
        displayed.append((uri, code))

    async def sleep(delay):
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        setup = PhotonSetup(client, sleep=sleep)
        await setup.login(display)
        first = await setup.setup(phone="+15551234567", create_project="Harness")
        second = await setup.setup(phone="+15551234567", create_project="Harness")
    assert first == second and first["user_id"] == "user"
    assert sleeps == [1, 6] and displayed == [(HOST + "/device", "ABCD-EFGH")]
    assert len(projects) == len(users) == 1
    assert not any("regenerate" in path for _, path, _ in requests)


async def test_photon_device_rejects_untrusted_destination_and_denied_login():
    replies = [
        {
            "device_code": "secret",
            "user_code": "CODE",
            "verification_uri": "https://attacker.example/device",
        },
        {"device_code": "secret", "user_code": "CODE", "verification_uri": HOST + "/device"},
    ]

    def http(request):
        if request.url.path.endswith("/code"):
            return httpx.Response(200, json=replies.pop(0))
        return httpx.Response(400, json={"error": "access_denied"})

    async def noop(*args):
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        setup = PhotonSetup(client, sleep=noop)
        with pytest.raises(ChannelError, match="destination"):
            await setup.login(noop)
        with pytest.raises(ChannelError, match="denied"):
            await setup.login(noop)


async def test_photon_private_file_explicit_factory_and_project_owner(tmp_path):
    account = {"project_id": "project", "project_secret": "never-print", "user_id": "owner"}
    target = save_photon_account(tmp_path, account)
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o600
    config = ChannelConfig(app_id="project", account_file=str(target), allowed_users=["owner"])
    transport = build_transport("photon", config)
    assert transport.token == "never-print"
    await transport.close()
    with pytest.raises(ChannelError, match="different project"):
        build_transport(
            "photon",
            ChannelConfig(app_id="other", account_file=str(target), allowed_users=["owner"]),
        )
    with pytest.raises(ValueError, match="exists"):
        save_photon_account(tmp_path, account)
    target.unlink()
    target.symlink_to(tmp_path / "victim")
    with pytest.raises(ValueError, match="symlinks"):
        photon_account_path(tmp_path)


def test_photon_setup_cli_redacts_secrets_and_emits_usable_configuration(tmp_path, monkeypatch):
    class FakeSetup:
        def __init__(self, client):
            pass

        async def login(self, display, **kwargs):
            await display(HOST + "/device", "PUBLIC-CODE")

        async def setup(self, **kwargs):
            return {
                "project_id": "project",
                "project_secret": "NEVER_PRINT",
                "user_id": "owner",
                "assigned_phone": "+15557654321",
            }

    monkeypatch.setattr(channel_photon_commands, "PhotonSetup", FakeSetup)
    app = typer.Typer()
    channel_photon_commands.register_photon_commands(app)
    result = CliRunner().invoke(
        app, ["--cwd", str(tmp_path), "--phone", "+15551234567", "--project", "project"]
    )
    assert result.exit_code == 0, result.output
    assert "NEVER_PRINT" not in result.output and 'allowed_users = ["owner"]' in result.output
    assert str(photon_account_path(tmp_path)) in result.output
