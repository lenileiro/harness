"""Tests for structured output (result_type=MyModel on RunRequest)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from harness.core import Done, RunRequest
from harness.core.schemas import Message

from .conftest import MockAdapter, MockStorage, text_turn


def make_agent(adapter, default_cwd="/tmp"):
    from harness.core import Agent, FailoverPolicy, ToolRegistry

    storage = MockStorage()
    registry = ToolRegistry()
    failover = FailoverPolicy(chain=[adapter.name], max_attempts=1)
    return Agent(
        adapters={adapter.name: adapter},
        tools=registry,
        storage=storage,
        failover=failover,
        default_model="test-model",
        default_cwd=default_cwd,
    )


async def collect(it):
    out = []
    async for e in it:
        out.append(e)
    return out


class SummaryModel(BaseModel):
    title: str
    score: int


@pytest.mark.asyncio
class TestStructuredOutput:
    async def test_valid_json_populates_structured_result(self, tmp_path) -> None:
        """When the model returns valid JSON, Done.structured_result is set."""
        payload = json.dumps({"title": "Report", "score": 9})
        adapter = MockAdapter(
            "mock",
            scripts=[
                [
                    Done(
                        final_message=Message(role="assistant", content=payload),
                    )
                ]
            ],
        )
        agent = make_agent(adapter, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="summarize", result_type=SummaryModel)))

        done = next((e for e in events if isinstance(e, Done)), None)
        assert done is not None
        assert done.structured_result == {"title": "Report", "score": 9}

    async def test_no_result_type_structured_result_is_none(self, tmp_path) -> None:
        """Without result_type, Done.structured_result stays None."""
        adapter = MockAdapter("mock", scripts=[text_turn("plain text answer")])
        agent = make_agent(adapter, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="hi")))

        done = next((e for e in events if isinstance(e, Done)), None)
        assert done is not None
        assert done.structured_result is None

    async def test_markdown_fenced_json_is_parsed(self, tmp_path) -> None:
        """JSON wrapped in ```json...``` fences is stripped and parsed."""
        payload = "```json\n" + json.dumps({"title": "Fenced", "score": 7}) + "\n```"
        adapter = MockAdapter(
            "mock",
            scripts=[[Done(final_message=Message(role="assistant", content=payload))]],
        )
        agent = make_agent(adapter, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="go", result_type=SummaryModel)))

        done = next((e for e in events if isinstance(e, Done)), None)
        assert done is not None
        assert done.structured_result == {"title": "Fenced", "score": 7}

    async def test_invalid_json_triggers_retry(self, tmp_path) -> None:
        """Invalid JSON causes a retry; the second valid response is used."""
        valid_payload = json.dumps({"title": "Retry", "score": 5})
        adapter = MockAdapter(
            "mock",
            scripts=[
                # First attempt: invalid JSON
                [Done(final_message=Message(role="assistant", content="not json at all"))],
                # Second attempt: valid JSON
                [Done(final_message=Message(role="assistant", content=valid_payload))],
            ],
        )
        agent = make_agent(adapter, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="summarize", result_type=SummaryModel)))

        done = next((e for e in events if isinstance(e, Done)), None)
        assert done is not None
        assert done.structured_result == {"title": "Retry", "score": 5}

    async def test_schema_injected_as_system_message(self, tmp_path) -> None:
        """The adapter receives a system message containing the JSON schema."""
        payload = json.dumps({"title": "Schema", "score": 3})
        adapter = MockAdapter(
            "mock",
            scripts=[[Done(final_message=Message(role="assistant", content=payload))]],
        )
        agent = make_agent(adapter, default_cwd=str(tmp_path))
        await collect(agent.run(RunRequest(prompt="hi", result_type=SummaryModel)))

        # The adapter should have been called with a system message containing the schema.
        assert adapter.calls, "expected at least one adapter call"
        messages = adapter.calls[0]["messages"]
        system_messages = [m for m in messages if m.role == "system"]
        schema_messages = [m for m in system_messages if "JSON" in (m.content or "")]
        assert schema_messages, "expected a system message with JSON schema instruction"
        assert "title" in schema_messages[0].content
