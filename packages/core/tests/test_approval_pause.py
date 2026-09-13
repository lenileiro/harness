import pytest

from harness.core import (
    Agent,
    ApprovalPolicy,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    RunRequest,
    ToolRegistry,
)
from harness.storage.memory import InMemoryStorage

from .conftest import MockAdapter, MockTool, text_turn, tool_call_turn


async def test_queued_action_pauses_without_another_model_call_or_verification():
    store = InMemoryStorage()
    tool = MockTool(name="shell", approval="prompt")
    registry = ToolRegistry()
    registry.register(tool)
    adapter = MockAdapter(
        "mock", scripts=[tool_call_turn(call_id="c", name="shell", arguments={"text": "act"})]
    )

    class NeverVerify:
        name = "never"

        async def verify(self, **kwargs):
            raise AssertionError("Approval pause must bypass verification and repair")

    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        failover=FailoverPolicy(chain=["mock"]),
        approval_store=store,
        approval_handler=InboxApprovalHandler(approval_store=store),
        pause_on_approval=True,
        verifier=NeverVerify(),
    )
    events = [
        event async for event in agent.run(RunRequest(prompt="act", model="m", session_id="s"))
    ]
    assert not tool.calls and len(adapter.calls) == 1
    assert len(await store.list_approvals(status="pending")) == 1
    session = await store.get("s")
    assert session is not None and session.status == "paused"
    assert any(
        isinstance(event, Done) and event.structured_result == {"status": "waiting_for_approval"}
        for event in events
    )


@pytest.mark.parametrize("deny_on_resume", [False, True])
async def test_replay_respects_current_policy_and_redacts_saved_results(deny_on_resume):
    store = InMemoryStorage()
    secret = "sk-" + "sensitive" * 5
    tool = MockTool(name="external", approval="prompt", responder=lambda **kwargs: secret)
    registry = ToolRegistry()
    registry.register(tool)
    adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(call_id="c", name="external", arguments={"text": "act"}),
            text_turn("done"),
        ],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        failover=FailoverPolicy(chain=["mock"]),
        approval_store=store,
        approval_handler=InboxApprovalHandler(approval_store=store),
        pause_on_approval=True,
    )
    async for _ in agent.run(RunRequest(prompt="act", model="m", session_id="s")):
        pass
    [approval] = await store.list_approvals(status="pending")
    await store.resolve_approval(approval.id, status="granted")
    if deny_on_resume:
        agent.approval_policy = ApprovalPolicy(per_tool={"external": "deny"})
    async for _ in agent.resume("s", prompt="continue"):
        pass
    saved = await store.get("s")
    assert saved is not None
    assert secret not in saved.model_dump_json()
    result = next(m.content for m in saved.messages if m.role == "tool")
    assert result is not None
    if deny_on_resume:
        assert not tool.calls and "denied by current policy" in result
    else:
        assert len(tool.calls) == 1 and "REDACTED" in result
