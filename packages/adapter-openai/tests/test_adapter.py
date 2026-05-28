"""OpenAIAdapter tests using httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from harness.adapters.openai import (
    OpenAIAdapter,
    inspect_codex_openai_auth,
    load_codex_openai_api_key,
)
from harness.core import (
    ConfigurationError,
    Done,
    Event,
    InternalError,
    Message,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TextDelta,
    ToolCallEvent,
)


def make_sse(*events: dict | str) -> bytes:
    lines = []
    for ev in events:
        body = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {body}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "index": 0}]}


def tool_chunk(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    args_fragment: str | None = None,
) -> dict:
    delta: dict = {"index": index}
    if call_id is not None:
        delta["id"] = call_id
    func: dict = {}
    if name is not None:
        func["name"] = name
    if args_fragment is not None:
        func["arguments"] = args_fragment
    if func:
        delta["function"] = func
    return {"choices": [{"delta": {"tool_calls": [delta]}, "index": 0}]}


async def collect(it) -> list[Event]:
    out: list[Event] = []
    async for e in it:
        out.append(e)
    return out


class TestConstruction:
    def test_missing_api_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            OpenAIAdapter()

    def test_env_key_is_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        adapter = OpenAIAdapter()
        assert adapter.api_key == "test-key"

    def test_codex_auth_api_key_is_used_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "codex-key"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        adapter = OpenAIAdapter()
        assert adapter.api_key == "codex-key"

    def test_codex_chatgpt_token_is_not_treated_as_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "OPENAI_API_KEY": None,
                    "tokens": {"access_token": "oauth-token"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        with pytest.raises(ConfigurationError, match="ChatGPT OAuth without OPENAI_API_KEY"):
            OpenAIAdapter()


class TestCodexAuthHelper:
    def test_returns_none_when_auth_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert load_codex_openai_api_key() is None

    def test_returns_none_for_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{oops", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert load_codex_openai_api_key() is None

    def test_inspect_reports_chatgpt_mode_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert inspect_codex_openai_auth() == {
            "auth_mode": "chatgpt",
            "has_openai_api_key": False,
        }


@pytest.mark.asyncio
class TestStream:
    async def test_text_response(self) -> None:
        body = make_sse(text_chunk("hi "), text_chunk("there"), "[DONE]")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
        ) as client:
            adapter = OpenAIAdapter(api_key="k", client=client)
            events = await collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="hi")])
            )
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["hi ", "there"]
        done = events[-1]
        assert isinstance(done, Done)
        assert done.final_message is not None
        assert done.final_message.content == "hi there"

    async def test_tool_call_accumulation(self) -> None:
        body = make_sse(
            tool_chunk(call_id="c1", name="ping"),
            tool_chunk(args_fragment='{"host":"x.com"}'),
            "[DONE]",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
        ) as client:
            adapter = OpenAIAdapter(api_key="k", client=client)
            events = await collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="ping x")])
            )
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].call.name == "ping"
        assert tool_events[0].call.arguments == {"host": "x.com"}


@pytest.mark.asyncio
class TestWireAndHeaders:
    async def test_uses_openai_base_url_and_auth_header(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAIAdapter(api_key="my-key", client=client)
            await collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="hi")])
            )

        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["headers"]["authorization"] == "Bearer my-key"


@pytest.mark.asyncio
class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, ConfigurationError),
            (404, ModelUnavailableError),
            (429, RateLimitError),
            (500, InternalError),
            (503, InternalError),
        ],
    )
    async def test_http_status_maps_to_typed_error(
        self, status: int, expected: type[Exception]
    ) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(status, content=b"err"))
        ) as client:
            adapter = OpenAIAdapter(api_key="k", client=client)
            with pytest.raises(expected):
                await collect(
                    adapter.stream(model="x", messages=[Message(role="user", content="hi")])
                )

    async def test_connect_error_maps_to_network(self) -> None:
        def raise_conn(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        async with httpx.AsyncClient(transport=httpx.MockTransport(raise_conn)) as client:
            adapter = OpenAIAdapter(api_key="k", client=client)
            with pytest.raises(NetworkError):
                await collect(
                    adapter.stream(model="x", messages=[Message(role="user", content="hi")])
                )
