"""Actual Codex binary, loopback fake model only; never uses real login/provider."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from harness.adapters.codex import CodexAdapter
from harness.core import Done, MediaAttachment, Message, ToolResult


async def test_installed_app_server_dynamic_rpc_and_native_tool_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binary = os.environ.get("HARNESS_CODEX_TEST_BINARY")
    if not binary:
        pytest.skip("set HARNESS_CODEX_TEST_BINARY to test an installed Codex 0.154+ binary")
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(json.loads(body))
            if len(requests) == 1:
                item = {
                    "type": "function_call",
                    "call_id": "c1",
                    "namespace": "harness",
                    "name": "lookup",
                    "arguments": '{"query":"test"}',
                }
            else:
                item = {
                    "type": "message",
                    "role": "assistant",
                    "id": "a1",
                    "content": [{"type": "output_text", "text": "Finished"}],
                }
            events = [
                {"type": "response.created", "response": {"id": "r1"}},
                {"type": "response.output_item.done", "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r1",
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                },
            ]
            data = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "auth.json").write_text('{"auth_mode":"apikey","OPENAI_API_KEY":"fake-only"}')
    sentinel = tmp_path / "native-mcp-started"
    (tmp_path / "config.toml").write_text(
        'model = "test-model"\nmodel_provider = "local_test"\n'
        '[model_providers.local_test]\nname = "Local test"\n'
        f'base_url = "http://127.0.0.1:{server.server_port}/v1"\n'
        'wire_api = "responses"\nrequires_openai_auth = false\n'
        "[features]\nenable_request_compression = false\n"
        '[mcp_servers.unwanted]\ncommand = "touch"\n'
        f"args = [{json.dumps(str(sentinel))}]\n"
    )
    adapter = CodexAdapter(codex_bin=binary, mode="app-server", timeout=20, idle_timeout=10)
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    png = MediaAttachment(
        kind="image",
        mime_type="image/png",
        data=(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/"
            "iZk9HQAAAABJRU5ErkJggg=="
        ),
    )
    messages = [Message(role="user", content="Call lookup", attachments=[png])]
    try:
        first = [
            e
            async for e in adapter.stream(
                model="test-model", messages=messages, tools=schemas, session_id="s"
            )
        ]
        final = next(e.final_message for e in first if isinstance(e, Done))
        assert final is not None and final.tool_calls
        call = final.tool_calls[0]
        result = ToolResult(
            tool_call_id=call.id, name=call.name, content="found", attachments=[png]
        )
        await adapter.submit_tool_result("s", result)
        messages.extend(
            [
                final,
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=result.content,
                    attachments=[png],
                ),
            ]
        )
        second = [
            e
            async for e in adapter.stream(
                model="test-model", messages=messages, tools=schemas, session_id="s"
            )
        ]
        final = next(e.final_message for e in second if isinstance(e, Done))
        assert final is not None and final.content == "Finished"
        assert len(requests) == 2
        for body in requests:
            assert [(tool.get("type"), tool.get("name")) for tool in body["tools"]] == [
                ("namespace", "harness")
            ]
        outputs = [item for item in requests[1]["input"] if item["type"] == "function_call_output"]
        assert outputs and outputs[-1]["call_id"] == call.id
        assert any(item["type"] == "input_image" for item in outputs[-1]["output"])
        assert not sentinel.exists()
        assert adapter._bridge is not None and not adapter._bridge.turns
    finally:
        await adapter.end_run("s")
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        await asyncio.to_thread(thread.join)
