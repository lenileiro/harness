import json

import httpx
import pytest
from typer.testing import CliRunner

from harness.cli.honcho_tools import HonchoConfig, HonchoToolset
from harness.core import Message, Session, ToolCall


@pytest.mark.asyncio
async def test_honcho_exports_only_selected_conversation_text_and_recalls_same_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HONCHO_API_KEY", "private-honcho-key")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer private-honcho-key"
        if request.url.path.endswith("/chat"):
            return httpx.Response(200, json={"content": "Prefers concise answers"})
        return httpx.Response(200, json=[] if request.url.path.endswith("/messages") else {})

    session = Session(
        provider="test",
        model="test",
        cwd=tmp_path / "work",
        messages=[
            Message(role="system", content="private instructions"),
            Message(role="user", content="I prefer concise answers"),
            Message(role="assistant", content="Understood"),
        ],
    )
    cfg = HonchoConfig(enabled=True)
    async with HonchoToolset(
        cfg, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        tools = {tool.name: tool for tool in owner.bind(session)}
        status = await tools["honcho_status"](ToolCall(id="s", name="honcho_status", arguments={}))
        assert not requests
        pending = json.loads(status.content)
        assert len(pending["messages"]) == 2
        refs = [message["reference"] for message in pending["messages"]]
        assert tools["honcho_sync"].approval == "prompt"
        exported = await tools["honcho_sync"](
            ToolCall(id="e", name="honcho_sync", arguments={"references": refs})
        )
        assert not exported.is_error
        recalled = await tools["honcho_chat"](
            ToolCall(id="q", name="honcho_chat", arguments={"query": "What preferences are known?"})
        )
        assert "concise" in recalled.content
    writes = [request for request in requests if request.url.path.endswith("/messages")]
    assert len(writes) == 1
    payload = json.loads(writes[0].content)
    assert [message["peer_id"] for message in payload["messages"]] == ["user", "assistant"]
    assert b"private instructions" not in writes[0].content
    async with HonchoToolset(
        cfg, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as restored:
        tool = next(tool for tool in restored.bind(session) if tool.name == "honcho_sync")
        duplicate = await tool(
            ToolCall(id="again", name="honcho_sync", arguments={"references": refs})
        )
        assert duplicate.is_error
    assert len([request for request in requests if request.url.path.endswith("/messages")]) == 1
    other = Session(provider="test", model="test", cwd=tmp_path / "different-workspace")
    async with HonchoToolset(
        cfg, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as restored:
        status_tool = restored.bind(other)[0]
        assert (
            json.loads(
                (await status_tool(ToolCall(id="s", name="honcho_status", arguments={}))).content
            )["workspace"]
            != pending["workspace"]
        )
        remote = Session(
            provider="test",
            model="test",
            cwd=session.cwd,
            metadata={"memory_scope": {"workspace": str(session.cwd), "user_id": "remote-person"}},
        )
        assert restored.bind(remote) == []


@pytest.mark.asyncio
async def test_ambiguous_honcho_write_requires_explicit_operator_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HONCHO_API_KEY", "secret")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HARNESS_PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    sent = []

    def respond(request):
        if request.url.path.endswith("/messages"):
            sent.append(request)
            raise httpx.ReadTimeout("response lost after commit")
        return httpx.Response(200, json={})

    cfg = HonchoConfig(enabled=True)
    session = Session(
        provider="test",
        model="test",
        cwd=tmp_path / "work",
        messages=[Message(role="user", content="Remember this preference")],
    )
    async with HonchoToolset(cfg, transport=httpx.MockTransport(respond)) as owner:
        tools = {tool.name: tool for tool in owner.bind(session)}
        status = json.loads(
            (
                await tools["honcho_status"](ToolCall(id="s", name="honcho_status", arguments={}))
            ).content
        )
        ref = status["messages"][0]["reference"]
        for number in range(2):
            result = await tools["honcho_sync"](
                ToolCall(id=str(number), name="honcho_sync", arguments={"references": [ref]})
            )
            assert result.is_error
        assert len(sent) == 1
        status = json.loads(
            (
                await tools["honcho_status"](ToolCall(id="s", name="honcho_status", arguments={}))
            ).content
        )
        assert status["messages"][0]["state"] == "uncertain"
    from harness.cli.__main__ import app

    result = CliRunner().invoke(
        app, ["honcho", "reconcile", status["workspace"], ref, "--outcome", "exported"]
    )
    assert result.exit_code == 0, result.output
    async with HonchoToolset(cfg, transport=httpx.MockTransport(respond)) as owner:
        status = json.loads(
            (
                await owner.bind(session)[0](ToolCall(id="s", name="honcho_status", arguments={}))
            ).content
        )
        assert status["messages"][0]["state"] == "exported"
    assert len(sent) == 1
