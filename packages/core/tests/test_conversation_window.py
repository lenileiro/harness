import json

from harness.core import Message, Session, ToolCall
from harness.core.memory import MemoryScope
from harness.core.session_search import message_references
from harness.core.tools_conversation_window import ConversationWindowTool

from .conftest import MockStorage


async def test_content_references_survive_appends_and_window_is_scoped(tmp_path):
    storage = MockStorage()
    session = Session(
        id="s",
        provider="p",
        model="m",
        cwd=tmp_path,
        messages=[
            Message(role="system", content="private system"),
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ],
    )
    before = message_references("s", session.messages)
    session.messages.append(Message(role="user", content="more"))
    assert message_references("s", session.messages)[: len(before)] == before
    await storage.save(session)
    tool = ConversationWindowTool(storage, scope=MemoryScope(workspace=str(tmp_path)))
    call = ToolCall(id="r", name=tool.name, arguments={"session_id": "s", "reference": before[1]})
    result = await tool(call)
    assert not result.is_error
    assert [m["content"] for m in json.loads(result.content)["messages"]] == [
        "question",
        "answer",
        "more",
    ]
    denied = await ConversationWindowTool(
        storage, scope=MemoryScope(workspace=str(tmp_path), user_id="different")
    )(call)
    assert denied.is_error and "question" not in denied.content
