import pytest

from harness.core import Capabilities, Message
from harness.core.compactor import ContextCompactor
from harness.core.events import Done


class FailedSummarizer:
    name = "test"

    async def capabilities(self):
        return Capabilities()

    async def cancel(self, session_id):
        return None

    def __init__(self, fail):
        self.fail = fail

    async def stream(self, **kwargs):
        if self.fail:
            raise RuntimeError("summarizer unavailable")
        yield Done(final_message=Message(role="assistant", content=""))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [True, False])
async def test_failed_or_empty_summary_preserves_original_transcript(fail):
    original = [
        Message(role="user", content="older facts " * 500),
        Message(role="assistant", content="prior answer " * 500),
        Message(role="user", content="latest request"),
    ]
    compactor = ContextCompactor(
        adapter=FailedSummarizer(fail), model="test", max_tokens=100, keep_recent_tokens=10
    )
    assert await compactor.compact(original) is original


@pytest.mark.asyncio
async def test_compaction_owns_separate_adapter_session_and_closes_stream():
    calls, closed, ended = [], [], []

    class OwnedSummarizer(FailedSummarizer):
        async def stream(self, **kwargs):
            calls.append(kwargs)
            try:
                yield Done(
                    final_message=Message(role="assistant", content="Prior work is complete.")
                )
            finally:
                closed.append(True)

        async def end_run(self, session_id):
            ended.append(session_id)

    original = [
        Message(role="user", content="old facts " * 1000),
        Message(role="user", content="current request"),
    ]
    compactor = ContextCompactor(
        adapter=OwnedSummarizer(False), model="test", keep_recent_tokens=10
    )
    for _ in range(2):
        assert await compactor.compact(original) is not original
    assert len(set(ended)) == 2 and closed == [True, True]
    assert [call["session_id"] for call in calls] == ended
    assert all(call["tools"] is None for call in calls)


@pytest.mark.asyncio
async def test_native_agent_summarization_cannot_execute_tools():
    class Native(FailedSummarizer):
        async def capabilities(self):
            return Capabilities(tool_use=True, external_tools=False)

        async def stream(self, **kwargs):
            pytest.fail("Native adapter must not receive the summary request")
            yield

    original = [Message(role="user", content="old " * 1000), Message(role="user", content="latest")]
    compactor = ContextCompactor(adapter=Native(False), model="test", keep_recent_tokens=10)
    assert await compactor.compact(original) is original


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["stream_close", "end_run"])
async def test_summary_cleanup_failure_preserves_history_and_attempts_owned_teardown(failure):
    ended = []

    class BrokenCleanup(FailedSummarizer):
        async def stream(self, **kwargs):
            try:
                yield Done(final_message=Message(role="assistant", content="Summary"))
            finally:
                if failure == "stream_close":
                    raise RuntimeError("Closing stream failed")

        async def end_run(self, session_id):
            ended.append(session_id)
            if failure == "end_run":
                raise RuntimeError("Owned process teardown failed")

    original = [
        Message(role="user", content="Old facts " * 1000),
        Message(role="user", content="Latest"),
    ]
    compactor = ContextCompactor(adapter=BrokenCleanup(False), model="test", keep_recent_tokens=10)
    assert await compactor.compact(original) is original
    assert len(ended) == 1
