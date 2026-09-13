import asyncio

from typer.testing import CliRunner

from harness.cli import channel_commands as commands
from harness.cli.config import HarnessConfig
from harness.core.gateway_channels import ChannelConfig


class Transport:
    def __init__(self, name):
        self.name = name
        self.closed = False

    async def close(self):
        self.closed = True


def test_run_all_cancels_siblings_and_closes_every_connection(monkeypatch, tmp_path):
    transports, started, cancelled = [], [], []
    cfg = HarnessConfig(
        channels={
            name: ChannelConfig(token_env="TEST", allowed_users=["owner"])
            for name in ["first", "second"]
        }
    )
    monkeypatch.setattr(commands, "load_config", lambda _: cfg)

    def build(name, config):
        result = Transport(name)
        transports.append(result)
        return result

    async def run(*, cwd, transport):
        started.append(transport.name)
        if transport.name == "first":
            await asyncio.sleep(0.01)
            raise RuntimeError("private credential-bearing transport error")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(transport.name)

    monkeypatch.setattr(commands, "build_transport", build)
    monkeypatch.setattr(commands, "run_transport", run)
    result = CliRunner().invoke(commands.channel_app, ["run-all", "--cwd", str(tmp_path)])
    assert result.exit_code == 1 and set(started) == {"first", "second"}
    assert cancelled == ["second"] and all(item.closed for item in transports)
    assert "private credential" not in result.output


def test_run_all_closes_preconstructed_clients_if_later_config_is_invalid(monkeypatch, tmp_path):
    first = Transport("first")
    cfg = HarnessConfig(
        channels={
            name: ChannelConfig(token_env="TEST", allowed_users=["owner"])
            for name in ["first", "second"]
        }
    )
    monkeypatch.setattr(commands, "load_config", lambda _: cfg)

    def build(name, config):
        if name == "first":
            return first
        raise ValueError("bad config")

    monkeypatch.setattr(commands, "build_transport", build)
    result = CliRunner().invoke(commands.channel_app, ["run-all", "--cwd", str(tmp_path)])
    assert result.exit_code == 1 and first.closed
