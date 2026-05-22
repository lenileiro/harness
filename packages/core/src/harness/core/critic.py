"""Critic: reviews failed repair attempts and challenges the agent's hypothesis.

A `Critic` is called by the repair loop after verification fails. Instead of
just handing the raw test output back to the agent, the critic produces a
targeted challenge: it identifies the mismatch between what the agent changed
and what the failing test actually checks, then asks the agent a pointed
question it has to answer before trying again.

This breaks the loop where a model re-applies the same wrong fix because it
treats the repair directive as "try harder," not "reconsider your premise."

When a `search_fn` is provided, `LLMCritic` runs a lightweight research pass
before generating its critique: it asks the LLM what to search for, calls the
search function, and incorporates the results. This lets the critic look up
relevant patterns (e.g. "Python asyncio concurrent deduplication") before
challenging the agent.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from harness.core.activity import ActivityEvent
from harness.core.adapter import Adapter
from harness.core.events import Done, TextDelta
from harness.core.schemas import Message, Session, VerificationResult


@runtime_checkable
class Critic(Protocol):
    """Reviews a failed repair attempt and returns a pointed critique.

    Called by the repair loop after verification fails. The returned string is
    prepended to the repair directive so the agent receives a specific challenge
    to address, not just raw failure output.

    Must not raise — return an empty string if critique cannot be produced.
    """

    async def critique(
        self,
        *,
        session: Session,
        verification_result: VerificationResult,
        activity: list[ActivityEvent],
    ) -> str: ...


SearchFn = Callable[[str], Awaitable[str]]

_RESEARCH_SYSTEM = """\
You are a code review critic preparing to challenge an AI agent's failed fix.

You have access to a web_search tool. Use it (1-2 calls max) to research \
relevant concepts that will help you write a more informed critique. Focus on:
- The specific pattern the failing test checks (e.g. deduplication, atomicity)
- Known correct implementations of that pattern in the relevant language

After searching, respond with JSON:
{"done": true, "searches_complete": true}

Or to search:
{"search": "your search query here"}
"""

_CRITIC_SYSTEM = """\
You are a code review critic embedded in an AI repair loop.

An AI agent attempted to fix a failing test suite, but the tests still fail.
Your job: identify the mismatch between what the agent changed and what the \
failing test actually checks.

Rules:
- Name the specific failing test and what it asserts
- Explain precisely why the agent's change does not address what that test checks
- Ask a pointed question the agent must answer before trying again
- Do NOT give away the correct solution — the agent must find it
- Be concise: 3-5 sentences maximum
- Tone: direct, not harsh. A senior engineer asking a junior "have you actually \
read what this test is checking?"

If the failure output is ambiguous or insufficient, say so in one sentence."""

_CRITIC_USER = """\
## Agent's last response

{agent_last}

## Test failure output

{failure}

{research_section}\
Write a 3-5 sentence critique identifying the mismatch between the agent's \
approach and what the failing test actually checks. Do not provide the solution.\
"""

_RESEARCH_SECTION = """\
## Web research findings

{findings}

"""


def _last_assistant_text(session: Session) -> str:
    for msg in reversed(session.messages):
        if msg.role == "assistant" and msg.content:
            return msg.content[:2000]
    return "(no assistant message found)"


async def _stream_text(adapter: Adapter, model: str, messages: list[Message], **kwargs) -> str:
    text_parts: list[str] = []
    async for event in adapter.stream(model=model, messages=messages, **kwargs):
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, Done):
            if event.final_message and event.final_message.content:
                return event.final_message.content.strip()
            break
    return "".join(text_parts).strip()


class LLMCritic:
    """Uses an LLM to critique failed repair attempts.

    Optionally runs a web research pass (1-2 searches) before generating the
    critique. Pass ``search_fn`` as a coroutine ``(query: str) -> str`` that
    returns search result text. ``TavilySearchTool`` can be wrapped this way.

    Args:
        adapter: Any harness Adapter for the critique call.
        model: Model identifier passed to the adapter.
        max_tokens: Cap on critique length (default 400).
        temperature: Sampling temperature (low = more focused critique).
        search_fn: Optional async callable for web research before critiquing.
        max_searches: Maximum search calls per critique (default 2).
    """

    def __init__(
        self,
        adapter: Adapter,
        model: str,
        *,
        max_tokens: int = 400,
        temperature: float = 0.3,
        search_fn: SearchFn | None = None,
        max_searches: int = 2,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._search_fn = search_fn
        self._max_searches = max_searches

    async def _research(self, agent_text: str, failure_text: str) -> str:
        """Run 1-2 searches and return concatenated findings."""
        messages: list[Message] = [
            Message(role="system", content=_RESEARCH_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Agent's last response:\n{agent_text[:1000]}\n\n"
                    f"Test failure:\n{failure_text[:1500]}\n\n"
                    "What should I search to better understand the failing test's intent? "
                    'Respond with {"search": "query"} or {"done": true}.'
                ),
            ),
        ]

        findings: list[str] = []
        for _ in range(self._max_searches):
            raw = await _stream_text(
                self._adapter, self._model, messages, max_tokens=100, temperature=0.0
            )
            try:
                obj = json.loads(raw.strip().strip("`").removeprefix("json").strip())
            except (json.JSONDecodeError, AttributeError):
                break

            if obj.get("done") or "search" not in obj:
                break

            query = str(obj["search"]).strip()
            if not query:
                break

            result = await self._search_fn(query)  # type: ignore[misc]
            findings.append(f"Search: {query}\n{result}")

            messages.append(Message(role="assistant", content=raw))
            messages.append(
                Message(role="user", content=f"Results:\n{result}\n\nContinue or {{'done': true}}.")
            )

        return "\n\n".join(findings)

    async def critique(
        self,
        *,
        session: Session,
        verification_result: VerificationResult,
        activity: list[ActivityEvent],
    ) -> str:
        agent_text = _last_assistant_text(session)
        failure_text = verification_result.reason[:3000]

        research_section = ""
        if self._search_fn is not None:
            try:
                findings = await self._research(agent_text, failure_text)
                if findings:
                    research_section = _RESEARCH_SECTION.format(findings=findings[:2000])
            except Exception:
                pass

        messages = [
            Message(role="system", content=_CRITIC_SYSTEM),
            Message(
                role="user",
                content=_CRITIC_USER.format(
                    agent_last=agent_text,
                    failure=failure_text,
                    research_section=research_section,
                ),
            ),
        ]

        try:
            return await _stream_text(
                self._adapter,
                self._model,
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            return ""


_DEVIL_SYSTEM = """\
You are a devil's advocate critic in an AI repair loop.

An AI agent's fix failed. Your role is different from the main critic: \
instead of explaining why the fix is wrong, you should suggest what the \
CORRECT approach likely looks like — without writing code.

Rules:
- Identify the design pattern or concept the failing test is checking \
  (e.g. "concurrent deduplication", "atomic compare-and-swap", "idempotency")
- In 1-2 sentences, hint at the right direction: what class of solution \
  addresses this pattern? (e.g. "in-flight request tracking using a dict of \
  futures/tasks", "a lock protecting shared state")
- Do NOT write code. Do NOT reproduce the solution. Just name the pattern \
  and one concrete hint about the data structure or mechanism.
- Be concise: 2-3 sentences maximum.
"""

_DEVIL_USER = """\
## Failing test output

{failure}

## Agent's last response

{agent_last}

Name the design pattern this test is checking and give one concrete hint \
about the mechanism needed (no code). 2-3 sentences.\
"""


class MultiCritic:
    """Runs two critic perspectives and concatenates their output.

    Critic 1 (``primary``): identifies the mismatch between the agent's change
    and what the failing test actually checks (hypothesis challenger).

    Critic 2 (``devil``): names the correct design pattern and hints at the
    mechanism without writing code (constructive nudge).

    Either or both may be ``None``; missing critics are silently skipped.
    """

    def __init__(self, primary: Critic, devil: Critic | None = None) -> None:
        self._primary = primary
        self._devil = devil

    async def critique(
        self,
        *,
        session: Session,
        verification_result: VerificationResult,
        activity: list[ActivityEvent],
    ) -> str:
        results: list[str] = []
        primary_text = await self._primary.critique(
            session=session,
            verification_result=verification_result,
            activity=activity,
        )
        if primary_text:
            results.append(f"**Critic 1 — hypothesis check:**\n{primary_text}")

        if self._devil is not None:
            devil_text = await self._devil.critique(
                session=session,
                verification_result=verification_result,
                activity=activity,
            )
            if devil_text:
                results.append(f"**Critic 2 — design pattern hint:**\n{devil_text}")

        return "\n\n".join(results)


class _DevilLLMCritic:
    """Devil's advocate: names the correct pattern and hints at the mechanism."""

    def __init__(
        self,
        adapter: Adapter,
        model: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.4,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def critique(
        self,
        *,
        session: Session,
        verification_result: VerificationResult,
        activity: list[ActivityEvent],
    ) -> str:
        agent_text = _last_assistant_text(session)
        failure_text = verification_result.reason[:3000]
        messages = [
            Message(role="system", content=_DEVIL_SYSTEM),
            Message(
                role="user",
                content=_DEVIL_USER.format(
                    failure=failure_text,
                    agent_last=agent_text[:1500],
                ),
            ),
        ]
        try:
            return await _stream_text(
                self._adapter,
                self._model,
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            return ""


def make_multi_critic(
    adapter: Adapter,
    model: str,
    *,
    search_fn: SearchFn | None = None,
) -> MultiCritic:
    """Build the default two-critic setup used by the CLI.

    Critic 1: hypothesis challenger with optional web search.
    Critic 2: devil's advocate pattern hint (no search needed).
    """
    primary = LLMCritic(adapter=adapter, model=model, search_fn=search_fn)
    devil = _DevilLLMCritic(adapter=adapter, model=model)
    return MultiCritic(primary=primary, devil=devil)


__all__ = ["Critic", "LLMCritic", "MultiCritic", "SearchFn", "make_multi_critic"]
