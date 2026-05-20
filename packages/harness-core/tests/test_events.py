from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from harness.core import (
    Done,
    ErrorEvent,
    Event,
    Message,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)

EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)


class TestEventDiscriminator:
    @pytest.mark.parametrize(
        ("payload", "cls"),
        [
            ({"type": "text_delta", "text": "hello"}, TextDelta),
            (
                {
                    "type": "tool_call",
                    "call": {"id": "c1", "name": "ping", "arguments": {}},
                },
                ToolCallEvent,
            ),
            (
                {
                    "type": "tool_result",
                    "result": {
                        "tool_call_id": "c1",
                        "name": "ping",
                        "content": "pong",
                        "is_error": False,
                    },
                },
                ToolResultEvent,
            ),
            ({"type": "step_started", "step": 0, "description": None}, StepStarted),
            ({"type": "step_completed", "step": 0}, StepCompleted),
            (
                {
                    "type": "done",
                    "final_message": {"role": "assistant", "content": "answer"},
                    "usage": None,
                },
                Done,
            ),
            (
                {
                    "type": "error",
                    "error": "boom",
                    "kind": "network",
                    "recoverable": True,
                },
                ErrorEvent,
            ),
        ],
    )
    def test_dispatches_to_correct_class(self, payload: dict, cls: type) -> None:
        ev = EventAdapter.validate_python(payload)
        assert isinstance(ev, cls)

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EventAdapter.validate_python({"type": "wat"})


class TestEventRoundTrip:
    def test_done_with_final_message(self) -> None:
        original = Done(
            final_message=Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
            )
        )
        wire = EventAdapter.dump_python(original, mode="json")
        decoded = EventAdapter.validate_python(wire)
        assert decoded == original

    def test_tool_result_event_round_trip(self) -> None:
        result = ToolResult(tool_call_id="c1", name="ping", content="pong")
        original = ToolResultEvent(result=result)
        wire = EventAdapter.dump_python(original, mode="json")
        decoded = EventAdapter.validate_python(wire)
        assert decoded == original
