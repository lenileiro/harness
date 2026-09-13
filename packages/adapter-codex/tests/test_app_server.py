from __future__ import annotations

# ruff: noqa: ASYNC240
import asyncio
import json
import sys
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from harness.adapters.codex import CodexAdapter
from harness.adapters.codex.app_server_protocol import response_items, tool_response
from harness.core import (
    Agent,
    ApprovalDecision,
    AutoApprove,
    ConfigurationError,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    MediaAttachment,
    Message,
    RunRequest,
    ToolCall,
    ToolCallEvent,
    ToolRegistry,
    ToolResult,
    ToolResultEvent,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError
from harness.storage.memory import InMemoryStorage

FAKE_SERVER = r"""
import json, os, sys, time
config = {}
args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == '-c':
        key, value = args[i+1].split('=', 1)
        cursor = config
        parts = key.split('.')
        for part in parts[:-1]: cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = json.loads(value)
log = os.environ['HARNESS_CODEX_TEST_LOG']
mode = os.environ.get('HARNESS_CODEX_TEST_MODE', 'tool')
def emit(obj):
    print(json.dumps(obj), flush=True)
def notify(method, params): emit({'method':method, 'params':params})
def finish():
    notify('item/agentMessage/delta', {'threadId':'thread1', 'itemId':'a', 'delta':'Complete'})
    notify('item/completed', {'threadId':'thread1', 'item':{'id':'a','type':'agentMessage','text':'Complete'}})
    notify('turn/completed', {'threadId':'thread1','turn':{'id':'turn1','status':'completed'}})
with open(log,'a') as f: f.write(json.dumps({'pid':os.getpid(),'cwd':os.getcwd(),'argv':args})+'\n')
injected = []
for line in sys.stdin:
    req = json.loads(line)
    with open(log,'a') as f: f.write(json.dumps(req)+'\n')
    method = req.get('method')
    if method == 'initialized': continue
    if method == 'initialize': result = {'userAgent':'fake-codex'}
    elif method == 'config/read':
        if mode == 'unsafe_config': config['features']['shell_tool'] = True
        result = {'config':config,'origins':{}}
    elif method == 'thread/start':
        result = {'thread':{'id':'thread1','environments':None if mode=='unsafe_env' else []}}
    elif method == 'thread/inject_items':
        injected = req['params']['items']; result = {}
    elif method == 'turn/start':
        emit({'id':req['id'],'result':{'turn':{'id':'turn1','status':'inProgress'}}})
        if mode == 'hang': time.sleep(60)
        elif mode == 'native':
            emit({'id':'native','method':'item/commandExecution/requestApproval','params':{}})
        elif any(item['type']=='function_call_output' for item in injected) or mode == 'text': finish()
        else:
            emit({'id':'request1','method':'item/tool/call','params':{'callId':'call1',
                'threadId':'thread1','turnId':'turn1','namespace':'harness','tool':'lookup',
                'arguments':{'query':'entry'}}})
        continue
    elif method == 'turn/interrupt': break
    elif req.get('id') == 'request1': finish(); continue
    else: result = {}
    emit({'id':req['id'],'result':result})
"""


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CodexAdapter:
    home = tmp_path / "auth"
    home.mkdir()
    (home / "auth.json").write_text('{"auth_mode":"apikey","OPENAI_API_KEY":"fake-test-only"}')
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("HARNESS_CODEX_TEST_LOG", str(tmp_path / "rpc.jsonl"))
    executable = tmp_path / "codex-fake"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_SERVER)
    executable.chmod(0o700)
    return CodexAdapter(codex_bin=str(executable), cwd=tmp_path, mode="app-server", idle_timeout=3)


def records(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "rpc.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a record",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


async def test_pending_rpc_roundtrip_preserves_result_status_and_media(
    adapter: CodexAdapter, tmp_path: Path
):
    messages = [
        Message(role="system", content="Trusted context"),
        Message(role="user", content="Find entry"),
    ]
    first = [
        event
        async for event in adapter.stream(
            model="openai/test", messages=messages, tools=TOOLS, session_id="s"
        )
    ]
    done = next(event for event in first if isinstance(event, Done))
    assert isinstance(first[0], ToolCallEvent)
    assert done.final_message is not None
    assert done.final_message.tool_calls and done.final_message.tool_calls[0].name == "lookup"
    image = MediaAttachment(kind="image", mime_type="image/png", data="aW1hZ2U=")
    result = ToolResult(
        tool_call_id="call1",
        name="lookup",
        content="Permission denied",
        is_error=True,
        attachments=[image],
    )
    await adapter.submit_tool_result("s", result)
    messages.extend(
        [
            done.final_message,
            Message(
                role="tool",
                tool_call_id="call1",
                name="lookup",
                content=result.content,
                attachments=[image],
            ),
        ]
    )
    second = [
        event
        async for event in adapter.stream(
            model="openai/test", messages=messages, tools=TOOLS, session_id="s"
        )
    ]
    final = next(event for event in second if isinstance(event, Done)).final_message
    assert final is not None and final.content == "Complete"
    trace = records(tmp_path)
    assert len([record for record in trace if "pid" in record]) == 1
    reply = next(record["result"] for record in trace if record.get("id") == "request1")
    assert reply == {
        "success": False,
        "contentItems": [
            {"type": "inputText", "text": "Permission denied"},
            {"type": "inputImage", "imageUrl": image.data_uri()},
        ],
    }
    thread = next(record["params"] for record in trace if record.get("method") == "thread/start")
    assert thread["environments"] == [] and thread["ephemeral"] is True
    assert thread["cwd"] != str(tmp_path) and not Path(thread["cwd"]).exists()
    assert not adapter._bridge.turns if adapter._bridge else False


async def test_context_or_model_change_rebuilds_without_replaying_effect(
    adapter: CodexAdapter, tmp_path: Path
):
    messages = [Message(role="user", content="Find entry")]
    first = [
        event
        async for event in adapter.stream(
            model="first", messages=messages, tools=TOOLS, session_id="s"
        )
    ]
    done = next(event for event in first if isinstance(event, Done))
    result = ToolResult(tool_call_id="call1", name="lookup", content="saved")
    assert done.final_message is not None
    await adapter.submit_tool_result("s", result)
    messages.extend(
        [
            done.final_message,
            Message(role="tool", tool_call_id="call1", name="lookup", content="saved"),
        ]
    )
    messages.insert(0, Message(role="system", content="Updated skill instructions"))
    second = [
        event
        async for event in adapter.stream(
            model="second", messages=messages, tools=TOOLS, session_id="s"
        )
    ]
    assert any(isinstance(event, Done) for event in second)
    trace = records(tmp_path)
    assert len([r for r in trace if "pid" in r]) == 2
    injection = [r["params"]["items"] for r in trace if r.get("method") == "thread/inject_items"][
        -1
    ]
    assert injection[0]["role"] == "developer"
    assert injection[-1]["type"] == "function_call_output" and injection[-1]["call_id"] == "call1"
    assert not any(r.get("id") == "request1" and "result" in r for r in trace)


@pytest.mark.parametrize("mode", ["unsafe_env", "unsafe_config", "native"])
async def test_native_escape_fails_closed(
    adapter: CodexAdapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
):
    monkeypatch.setenv("HARNESS_CODEX_TEST_MODE", mode)
    with pytest.raises(ConfigurationError):
        _ = [
            event
            async for event in adapter.stream(
                model="test", messages=[Message(role="user", content="x")], session_id="s"
            )
        ]
    assert adapter._bridge is not None and not adapter._bridge.turns
    assert not Path(records(tmp_path)[0]["cwd"]).exists()


async def test_timeout_and_cancellation_close_owned_process(
    adapter: CodexAdapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HARNESS_CODEX_TEST_MODE", "hang")
    assert adapter._bridge is not None
    adapter._bridge.idle_timeout = 0.1
    with pytest.raises(HarnessTimeoutError):
        _ = [event async for event in adapter.stream(model="test", messages=[], session_id="s")]
    assert not adapter._bridge.turns
    adapter._bridge.idle_timeout = 30
    previous = len([r for r in records(tmp_path) if r.get("method") == "turn/start"])

    async def collect():
        return [event async for event in adapter.stream(model="test", messages=[], session_id="s")]

    task = asyncio.create_task(collect())
    try:
        async with asyncio.timeout(5):
            while (
                len([r for r in records(tmp_path) if r.get("method") == "turn/start"]) <= previous
            ):
                if task.done():
                    await task
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not adapter._bridge.turns


@pytest.mark.parametrize("field,value", [("temperature", 0.2), ("max_tokens", 128)])
async def test_unsupported_sampling_is_explicit(
    adapter: CodexAdapter, tmp_path: Path, field: str, value: Any
):
    with pytest.raises(ConfigurationError, match="does not expose"):
        _ = [
            event
            async for event in adapter.stream(
                model="test", messages=[], session_id="s", **{field: value}
            )
        ]
    assert not (tmp_path / "rpc.jsonl").exists()


def test_transcript_preserves_roles_media_and_rejects_remote_urls():
    audio = MediaAttachment(kind="audio", mime_type="audio/wav", data="YXVkaW8=")
    items = response_items([Message(role="user", content="Listen", attachments=[audio])])
    assert items[0]["content"][1] == {"type": "input_audio", "audio_url": audio.data_uri()}
    remote = MediaAttachment(
        kind="image", mime_type="image/png", url="https://example.com/picture.png"
    )
    with pytest.raises(ConfigurationError, match="inline"):
        response_items([Message(role="user", attachments=[remote])])
    deliverable = remote.model_copy(update={"model_visible": False})
    assert tool_response(
        ToolResult(tool_call_id="c", name="t", content="saved", attachments=[deliverable])
    )["contentItems"] == [{"type": "inputText", "text": "saved"}]


class LookupTool:
    name = "lookup"
    description = "Look up a record"
    approval: ApprovalDecision = "prompt"

    def __init__(self) -> None:
        self.parameters_schema = TOOLS[0]["function"]["parameters"]
        self.calls = 0

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.calls += 1
        return ToolResult(tool_call_id=call.id, name=self.name, content="sk-" + "sensitive" * 5)


async def test_runtime_approval_evidence_redaction_and_cleanup(
    adapter: CodexAdapter, tmp_path: Path
):
    tool = LookupTool()
    registry = ToolRegistry()
    registry.register(tool)
    store = InMemoryStorage()
    agent = Agent(
        adapters={"codex": adapter},
        tools=registry,
        storage=store,
        activity_store=store,
        approval_handler=AutoApprove(),
        failover=FailoverPolicy(chain=["codex"], max_attempts=1),
    )
    events = [
        event
        async for event in agent.run(RunRequest(model="test", prompt="Find entry", session_id="s"))
    ]
    assert tool.calls == 1 and any(isinstance(event, ToolResultEvent) for event in events)
    reply = next(record["result"] for record in records(tmp_path) if record.get("id") == "request1")
    assert "REDACTED" in reply["contentItems"][0]["text"]
    assert "sensitive" not in json.dumps(reply)
    evidence = await store.list_activity(session_id="s")
    assert len([e for e in evidence if e.kind == "tool_call.completed"]) == 1
    assert adapter._bridge is not None and not adapter._bridge.turns


async def test_approval_pause_and_generator_close_release_rpc(adapter: CodexAdapter):
    store = InMemoryStorage()
    tool = LookupTool()
    registry = ToolRegistry()
    registry.register(tool)
    agent = Agent(
        adapters={"codex": adapter},
        tools=registry,
        storage=store,
        approval_store=store,
        approval_handler=InboxApprovalHandler(approval_store=store),
        pause_on_approval=True,
        failover=FailoverPolicy(chain=["codex"], max_attempts=1),
    )
    _ = [
        event
        async for event in agent.run(RunRequest(model="test", prompt="Find entry", session_id="s"))
    ]
    assert not tool.calls
    assert adapter._bridge is not None and not adapter._bridge.turns
    saved = await store.get("s")
    assert saved is not None and saved.status == "paused"
    async with aclosing(
        agent.run(RunRequest(model="test", prompt="Find entry", session_id="close"))
    ) as stream:
        async for event in stream:
            if isinstance(event, ToolCallEvent):
                break
    assert not adapter._bridge.turns


async def test_cancellation_during_process_creation_reaps_late_child(
    adapter: CodexAdapter, monkeypatch: pytest.MonkeyPatch
):
    original = asyncio.create_subprocess_exec
    created = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def delayed_create(*args: Any, **kwargs: Any):
        process = await original(*args, **kwargs)
        processes.append(process)
        created.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)

    async def collect():
        return [event async for event in adapter.stream(model="test", messages=[], session_id="s")]

    task = asyncio.create_task(collect())
    await asyncio.wait_for(created.wait(), 3)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None
    assert adapter._bridge is not None and not adapter._bridge.turns
