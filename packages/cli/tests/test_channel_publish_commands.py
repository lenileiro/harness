import json

import httpx
import typer
from typer.testing import CliRunner

from harness.cli import channel_publish_commands as commands
from harness.cli.channels.ntfy import NtfyTransport
from harness.cli.config import HarnessConfig
from harness.core.gateway_channels import ChannelConfig


def test_ntfy_publisher_emits_signed_input_not_secret_or_prompt(monkeypatch):
    config = ChannelConfig(
        homeserver="https://ntfy.example",
        topic="input",
        reply_topic="output",
        username="owner",
        allowed_users=["owner"],
    )
    monkeypatch.setattr(
        commands, "load_config", lambda path: HarnessConfig(channels={"ntfy": config})
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "published"})

    transport = NtfyTransport(
        config=config,
        token="access-secret",
        app_token="s" * 32,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(commands, "build_transport", lambda name, config: transport)
    app = typer.Typer()
    commands.register_publish_commands(app)
    result = CliRunner().invoke(app, ["private prompt"])
    assert result.exit_code == 0, result.output
    assert "private prompt" not in result.output and "secret" not in result.output
    output = json.loads(result.output)
    payload = json.loads(requests[0].content)
    assert payload["topic"] == "input"
    envelope = json.loads(payload["message"])
    assert envelope["payload"]["text"] == "private prompt"
    assert envelope["payload"]["id"] == output["message_id"]
    assert envelope["signature"] and "s" * 32 not in payload["message"]
