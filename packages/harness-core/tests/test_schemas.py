from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.core import (
    Capabilities,
    Message,
    RunRequest,
    Session,
    ToolCall,
    ToolResult,
    Usage,
)


class TestMessage:
    def test_user_message_minimal(self) -> None:
        m = Message(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_assistant_with_tool_calls(self) -> None:
        m = Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="ping", arguments={"host": "x.com"})],
        )
        assert m.tool_calls is not None
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "ping"

    def test_tool_message_round_trip(self) -> None:
        m = Message(role="tool", tool_call_id="c1", name="ping", content="pong")
        round = Message.model_validate(m.model_dump())
        assert round == m

    def test_unknown_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="other", content="x")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Message.model_validate({"role": "user", "content": "hi", "unexpected": True})


class TestToolCall:
    def test_arguments_default_empty(self) -> None:
        tc = ToolCall(id="c1", name="noop")
        assert tc.arguments == {}

    def test_arguments_dict_preserved(self) -> None:
        tc = ToolCall(id="c1", name="ping", arguments={"host": "x.com", "n": 3})
        assert tc.arguments == {"host": "x.com", "n": 3}


class TestToolResult:
    def test_is_error_defaults_false(self) -> None:
        tr = ToolResult(tool_call_id="c1", name="x", content="ok")
        assert tr.is_error is False

    def test_error_result(self) -> None:
        tr = ToolResult(tool_call_id="c1", name="x", content="boom", is_error=True)
        assert tr.is_error is True


class TestCapabilities:
    def test_defaults(self) -> None:
        c = Capabilities()
        assert c.streaming is True
        assert c.tool_use is False
        assert c.structured_output is False
        assert c.max_context_tokens is None
        assert c.models is None


class TestUsage:
    def test_zero_defaults(self) -> None:
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


class TestRunRequest:
    def test_minimal_prompt(self) -> None:
        r = RunRequest(prompt="hi")
        assert r.prompt == "hi"
        assert r.session_id.startswith("sess_")
        assert r.max_steps == 25
        assert r.stream is True

    def test_session_id_is_unique(self) -> None:
        a = RunRequest(prompt="a")
        b = RunRequest(prompt="b")
        assert a.session_id != b.session_id

    def test_overrides_carry_through(self) -> None:
        r = RunRequest(
            prompt="hi",
            session_id="sess_fixed",
            provider="ollama",
            model="llama3.2",
            temperature=0.4,
            max_tokens=200,
            max_steps=5,
        )
        assert r.session_id == "sess_fixed"
        assert r.provider == "ollama"
        assert r.model == "llama3.2"
        assert r.temperature == 0.4
        assert r.max_tokens == 200
        assert r.max_steps == 5


class TestSession:
    def test_session_creates_with_defaults(self, tmp_path: Path) -> None:
        s = Session(provider="ollama", model="llama3.2", cwd=tmp_path)
        assert s.id.startswith("sess_")
        assert s.status == "pending"
        assert s.messages == []
        assert s.cwd == tmp_path
        # Both default factories fire essentially simultaneously; allow microsecond skew.
        assert (s.updated_at - s.created_at).total_seconds() < 0.001

    def test_touch_bumps_updated_at(self, tmp_path: Path) -> None:
        s = Session(provider="ollama", model="llama3.2", cwd=tmp_path)
        before = s.updated_at
        s.touch()
        assert s.updated_at >= before

    def test_session_serializes_round_trip(self, tmp_path: Path) -> None:
        s = Session(
            provider="ollama",
            model="llama3.2",
            cwd=tmp_path,
            messages=[Message(role="user", content="hi")],
        )
        round = Session.model_validate(s.model_dump(mode="json"))
        assert round.id == s.id
        assert round.provider == s.provider
        assert round.cwd == s.cwd
        assert round.messages == s.messages
