import json
from urllib.parse import parse_qs

import httpx
import pytest

from harness.cli.account_auth import AccountAuth, OAuthAccountConfig
from harness.core import ConfigurationError


def config(**changes):
    return OAuthAccountConfig(
        device_authorization_endpoint="https://login.example/device",
        token_endpoint="https://login.example/token",
        client_id="harness-test",
        scopes="inference:invoke",
        **changes,
    )


@pytest.mark.asyncio
async def test_device_poll_interval_slowdown_refresh_rotation_and_restart(tmp_path):
    requests, displays, sleeps = [], [], []
    clock = [0]
    attempts = [0]

    def respond(request):
        requests.append(request)
        data = parse_qs(request.content.decode())
        if request.url.path == "/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "private-device-code",
                    "user_code": "USER-CODE",
                    "verification_uri": "https://login.example/verify",
                    "expires_in": 120,
                    "interval": 2,
                },
            )
        if data["grant_type"] == ["refresh_token"]:
            assert data["refresh_token"] == ["refresh-1"]
            return httpx.Response(
                200,
                json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
            )
        attempts[0] += 1
        if attempts[0] == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        if attempts[0] == 2:
            return httpx.Response(400, json={"error": "slow_down"})
        return httpx.Response(
            200, json={"access_token": "access-1", "refresh_token": "refresh-1", "expires_in": 3600}
        )

    transport = httpx.MockTransport(respond)
    identity = AccountAuth(
        "example", config(), resource="https://model.example/v1", home=tmp_path, transport=transport
    )

    async def display(url, code):
        displays.append((url, code))

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    await identity.login(display, sleep=sleep, monotonic=lambda: clock[0])
    assert sleeps == [2, 2, 7]
    assert displays == [("https://login.example/verify", "USER-CODE")]
    assert "private-device-code" not in identity.path.read_text()
    assert await identity.access_token() == "access-1"
    state = json.loads(identity.path.read_text())
    state["expires_at"] = 0
    identity.path.write_text(json.dumps(state))
    restored = AccountAuth(
        "example", config(), resource="https://model.example/v1", home=tmp_path, transport=transport
    )
    assert await restored.access_token() == "access-2"
    assert json.loads(restored.path.read_text())["refresh_token"] == "refresh-2"
    assert all("access-" not in str(value) for value in restored.status().values())
    assert len(requests) == 5


@pytest.mark.asyncio
async def test_terminal_refresh_failure_is_quarantined_and_bound_to_resource(tmp_path):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            400, json={"error": "invalid_grant", "error_description": "private-secret"}
        )

    auth = AccountAuth(
        "example",
        config(refresh_token_header="x-test-refresh"),
        resource="https://model.example/v1",
        home=tmp_path,
        transport=httpx.MockTransport(respond),
    )
    auth._store_login({"access_token": "access", "refresh_token": "refresh", "expires_at": 0})
    with pytest.raises(ConfigurationError, match="refresh rejected"):
        await auth.access_token()
    assert auth.status()["quarantined"]
    assert seen[0].headers["x-test-refresh"] == "refresh"
    assert b"refresh_token=refresh" not in seen[0].content
    with pytest.raises(ConfigurationError, match="login required"):
        await auth.access_token()
    assert len(seen) == 1
    other = AccountAuth(
        "example",
        config(refresh_token_header="x-test-refresh"),
        resource="https://different.example/v1",
        home=tmp_path,
        transport=httpx.MockTransport(respond),
    )
    assert not other.status()["configured"]
    assert other.path != auth.path


@pytest.mark.asyncio
async def test_refresh_serializes_across_account_instances(tmp_path):
    import asyncio
    import time

    seen = []

    def respond(request):
        seen.append(request)
        time.sleep(0.03)
        return httpx.Response(
            200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600}
        )

    first = AccountAuth(
        "example",
        config(),
        resource="https://model.example/v1",
        home=tmp_path,
        transport=httpx.MockTransport(respond),
    )
    second = AccountAuth(
        "example",
        config(),
        resource="https://model.example/v1",
        home=tmp_path,
        transport=httpx.MockTransport(respond),
    )
    first._store_login({"access_token": "access", "refresh_token": "refresh", "expires_at": 0})
    assert await asyncio.gather(first.access_token(), second.access_token()) == ["fresh", "fresh"]
    assert len(seen) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"expires_in": float("nan")},
        {"expires_in": float("inf")},
        {"expires_in": True},
        {"token_type": 123},
    ],
)
def test_malformed_credentials_are_rejected_without_logging_tokens(tmp_path, changes):
    auth = AccountAuth("example", config(), resource="https://model.example/v1", home=tmp_path)
    with pytest.raises(ConfigurationError) as error:
        auth._tokens({"access_token": "private-secret", "expires_in": 3600, **changes})
    assert "private-secret" not in str(error.value)
    assert not auth.path.exists()
