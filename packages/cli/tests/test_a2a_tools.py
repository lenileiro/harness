import asyncio
import json

import httpx
import pytest

from harness.cli.a2a_tools import A2AConfig, A2APeer, A2AToolset
from harness.core import (
    Agent,
    Capabilities,
    Done,
    FailoverPolicy,
    Message,
    Session,
    ToolCall,
    ToolRegistry,
)
from harness.server import HarnessService, ServerAuth, create_app


class Adapter:
    name = "fixture"

    def __init__(self):
        self.calls = []

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield Done(final_message=Message(role="assistant", content="Completed peer work"))


def tool(owner, session, name):
    return next(item for item in owner.bind(session) if item.name == name)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["1.0", "0.3"])
async def test_peers_roundtrip_official_sdk_server_and_resume_scoped_history(
    tmp_path, monkeypatch, protocol
):
    secret = "private-peer-token-" + "t" * 40
    monkeypatch.setenv("PEER_TOKEN", secret)
    adapter = Adapter()

    def builder(context):
        return Agent(
            adapters={"fixture": adapter},
            tools=ToolRegistry(),
            storage=context.storage,
            failover=FailoverPolicy(chain=["fixture"]),
            default_model="test",
            default_cwd=str(context.workspace),
        )

    remote = tmp_path / "remote"
    remote.mkdir()
    service = HarnessService(tmp_path / "server.db", remote, builder)
    app = create_app(
        service,
        auth=ServerAuth(token_envs={"caller": "PEER_TOKEN"}),
        a2a_url="http://localhost/a2a",
    )
    config = A2AConfig(
        enabled=True,
        poll_interval=0.05,
        peers={
            "researcher": A2APeer(
                url="http://localhost/a2a",
                token_env="PEER_TOKEN",
                protocol=protocol,
                timeout=5,
                capabilities=("research",),
            )
        },
    )
    session = Session(provider="fixture", model="test", cwd=tmp_path / "local")
    async with app.router.lifespan_context(app):
        async with A2AToolset(
            config, home=tmp_path / "home", transport=httpx.ASGITransport(app=app)
        ) as owner:
            called = await tool(owner, session, "a2a_call")(
                ToolCall(
                    id="first",
                    name="a2a_call",
                    arguments={"peer": "researcher", "message": "Do the work. Token " + secret},
                )
            )
            assert not called.is_error, called.content
            first = json.loads(called.content)
            assert first["state"] == "completed" and first["text"] == "Completed peer work"
            assert len(adapter.calls) == 1
            assert secret not in str(adapter.calls)
            context = first["context_id"]
            rejected = await tool(
                owner,
                Session(provider="fixture", model="test", cwd=tmp_path / "other"),
                "a2a_history",
            )(ToolCall(id="history", name="a2a_history", arguments={"context_id": context}))
            assert rejected.is_error and "Completed peer work" not in rejected.content
        async with A2AToolset(
            config, home=tmp_path / "home", transport=httpx.ASGITransport(app=app)
        ) as restarted:
            history = await tool(restarted, session, "a2a_history")(
                ToolCall(id="history", name="a2a_history", arguments={"context_id": context})
            )
            assert not history.is_error and secret not in history.content
            assert json.loads(history.content)[0]["state"] == "completed"
            duplicate = await tool(restarted, session, "a2a_call")(
                ToolCall(
                    id="first",
                    name="a2a_call",
                    arguments={"peer": "researcher", "message": "Do the work. Token " + secret},
                )
            )
            assert duplicate.is_error and len(adapter.calls) == 1
            continued = await tool(restarted, session, "a2a_call")(
                ToolCall(
                    id="second",
                    name="a2a_call",
                    arguments={
                        "peer": "researcher",
                        "message": "Continue the same work",
                        "context_id": context,
                    },
                )
            )
            assert not continued.is_error, continued.content
            assert (
                json.loads(continued.content)["context_id"] == context and len(adapter.calls) == 2
            )
            assert any(
                message.content == "Continue the same work"
                for message in adapter.calls[-1]["messages"]
            )
            assert (
                len(
                    json.loads(
                        (
                            await tool(restarted, session, "a2a_history")(
                                ToolCall(
                                    id="h", name="a2a_history", arguments={"context_id": context}
                                )
                            )
                        ).content
                    )
                )
                == 2
            )


@pytest.mark.asyncio
async def test_advertised_endpoint_cannot_receive_peer_token_on_another_origin(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PEER_TOKEN", "private-peer-secret")
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "supportedInterfaces": [
                    {
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                        "url": "https://other.example/a2a",
                    }
                ]
            },
        )

    config = A2AConfig(
        enabled=True, peers={"peer": A2APeer(url="https://peer.example", token_env="PEER_TOKEN")}
    )
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        result = await tool(owner, session, "a2a_call")(
            ToolCall(id="send", name="a2a_call", arguments={"peer": "peer", "message": "work"})
        )
        assert result.is_error and "origin" in result.content
    assert len(seen) == 1 and seen[0].url.host == "peer.example"


@pytest.mark.asyncio
async def test_uncertain_submission_survives_restart_without_implicit_retry(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"url": "https://peer.example/a2a"})
        raise httpx.ReadError("response lost")

    config = A2AConfig(enabled=True, peers={"peer": A2APeer(url="https://peer.example")})
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    call = ToolCall(id="same", name="a2a_call", arguments={"peer": "peer", "message": "work"})
    for _ in range(2):
        async with A2AToolset(
            config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
        ) as owner:
            result = await tool(owner, session, "a2a_call")(call)
            assert result.is_error
            listed = await tool(owner, session, "a2a_list")(
                ToolCall(id="list", name="a2a_list", arguments={})
            )
            contexts = json.loads(listed.content)["conversations"]
            assert len(contexts) == 1 and contexts[0]["state"] == "uncertain"
    assert sum(request.method == "POST" for request in requests) == 1


@pytest.mark.asyncio
async def test_capability_orchestration_is_bounded_and_keeps_every_exchange(tmp_path):
    active, max_active = [0], [0]

    async def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"url": str(request.url.copy_with(path="/a2a"))})
        active[0] += 1
        max_active[0] = max(max_active[0], active[0])
        try:
            await asyncio.sleep(0.02)
            data = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": data["id"],
                    "result": {
                        "message": {
                            "messageId": "answer",
                            "contextId": "remote-" + request.url.host,
                            "parts": [{"text": "Answer from " + request.url.host}],
                        }
                    },
                },
            )
        finally:
            active[0] -= 1

    config = A2AConfig(
        enabled=True,
        max_parallel=2,
        peers={
            name: A2APeer(url=f"https://{name}.example", capabilities=("research",))
            for name in ["one", "two", "three"]
        },
    )
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        result = await tool(owner, session, "a2a_orchestrate")(
            ToolCall(
                id="fanout",
                name="a2a_orchestrate",
                arguments={"capability": "research", "message": "Work", "mode": "all"},
            )
        )
        assert not result.is_error, result.content
        values = json.loads(result.content)["results"]
        assert len(values) == 3 and all(item["state"] == "completed" for item in values)
        assert max_active[0] == 2
        listed = json.loads(
            (
                await tool(owner, session, "a2a_list")(
                    ToolCall(id="list", name="a2a_list", arguments={})
                )
            ).content
        )
        assert len(listed["conversations"]) == 3


@pytest.mark.asyncio
async def test_peer_submission_waits_for_persisted_approval(tmp_path):
    from harness.core import InboxApprovalHandler, RunRequest
    from harness.storage.memory import InMemoryStorage

    requests = []

    class CallingAdapter(Adapter):
        async def stream(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="remote",
                                name="a2a_call",
                                arguments={"peer": "peer", "message": "Perform work"},
                            )
                        ],
                    )
                )
            else:
                yield Done(final_message=Message(role="assistant", content="Finished"))

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"url": "https://peer.example/a2a"})
        data = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": data["id"],
                "result": {"parts": [{"text": "Done"}], "contextId": "remote"},
            },
        )

    config = A2AConfig(enabled=True, peers={"peer": A2APeer(url="https://peer.example")})
    storage = InMemoryStorage()
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:

        async def bind(session):
            return owner.bind(session)

        agent = Agent(
            adapters={"fixture": CallingAdapter()},
            tools=ToolRegistry(),
            storage=storage,
            failover=FailoverPolicy(chain=["fixture"]),
            default_model="test",
            default_cwd=str(tmp_path),
            approval_store=storage,
            approval_handler=InboxApprovalHandler(approval_store=storage),
            pause_on_approval=True,
            session_tool_factory=bind,
        )
        events = [
            event async for event in agent.run(RunRequest(prompt="Ask the peer", session_id="s"))
        ]
        assert requests == []
        assert any(
            isinstance(event, Done)
            and event.structured_result == {"status": "waiting_for_approval"}
            for event in events
        )
        [approval] = await storage.list_approvals(status="pending")
        assert approval.tool_name == "a2a_call"
        await storage.resolve_approval(approval.id, status="granted")
        async for _ in agent.resume("s", prompt="Continue"):
            pass
        assert sum(request.method == "POST" for request in requests) == 1


@pytest.mark.asyncio
async def test_known_task_timeout_can_be_inspected_and_cancelled_after_restart(tmp_path):
    methods = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"url": "https://peer.example/a2a"})
        data = json.loads(request.content)
        methods.append(data["method"])
        state = "TASK_STATE_CANCELED" if data["method"] == "CancelTask" else "TASK_STATE_WORKING"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": data["id"],
                "result": {
                    "task": {"id": "task", "contextId": "context", "status": {"state": state}}
                },
            },
        )

    config = A2AConfig(
        enabled=True,
        poll_interval=0.05,
        peers={"peer": A2APeer(url="https://peer.example", timeout=0.08)},
    )
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        response = await tool(owner, session, "a2a_call")(
            ToolCall(id="send", name="a2a_call", arguments={"peer": "peer", "message": "work"})
        )
        assert not response.is_error, response.content
        saved = json.loads(response.content)
        assert saved["state"] == "working" and "still active" in saved["notice"]
        context = saved["context_id"]
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        status = await tool(owner, session, "a2a_status")(
            ToolCall(id="status", name="a2a_status", arguments={"context_id": context})
        )
        assert json.loads(status.content)["state"] == "working"
        cancelled = await tool(owner, session, "a2a_cancel")(
            ToolCall(id="cancel", name="a2a_cancel", arguments={"context_id": context})
        )
        assert json.loads(cancelled.content)["state"] == "cancelled"
    assert methods.count("SendMessage") == 1 and methods.count("CancelTask") == 1


@pytest.mark.asyncio
async def test_input_required_continues_task_but_new_uncertain_turn_does_not_reuse_old_handle(
    tmp_path,
):
    sends = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"url": "https://peer.example/a2a"})
        data = json.loads(request.content)
        sends.append(data)
        if len(sends) == 3:
            raise httpx.ReadError("lost")
        state = "TASK_STATE_INPUT_REQUIRED" if len(sends) == 1 else "TASK_STATE_COMPLETED"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": data["id"],
                "result": {
                    "task": {
                        "id": "task",
                        "contextId": "context",
                        "status": {"state": state, "message": {"parts": [{"text": "Reply"}]}},
                    }
                },
            },
        )

    config = A2AConfig(enabled=True, peers={"peer": A2APeer(url="https://peer.example")})
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        context = None
        for i in range(3):
            args = {"peer": "peer", "message": "Continue"}
            if context:
                args["context_id"] = context
            response = await tool(owner, session, "a2a_call")(
                ToolCall(id=str(i), name="a2a_call", arguments=args)
            )
            if i < 2:
                assert not response.is_error, response.content
                context = json.loads(response.content)["context_id"]
            else:
                assert response.is_error
        status = await tool(owner, session, "a2a_status")(
            ToolCall(id="status", name="a2a_status", arguments={"context_id": context})
        )
        assert json.loads(status.content)["state"] == "uncertain"
        assert len(sends) == 3
    assert sends[1]["params"]["message"]["taskId"] == "task"
    assert "taskId" not in sends[2]["params"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 401])
async def test_legacy_card_fallback_only_for_missing_new_card(tmp_path, status):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path.endswith("agent-card.json"):
            return httpx.Response(status, json={})
        return httpx.Response(200, json={"url": "https://peer.example/a2a"})

    config = A2AConfig(enabled=True, peers={"peer": A2APeer(url="https://peer.example")})
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        response = await tool(owner, session, "a2a_discover")(
            ToolCall(id="card", name="a2a_discover", arguments={"peer": "peer"})
        )
        assert response.is_error == (status == 401)
    assert len(paths) == (2 if status == 404 else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["first", "best"])
async def test_orchestration_selection_and_cancelled_waits_keep_remote_handles(tmp_path, mode):
    requests = []
    slow_started = asyncio.Event()

    async def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"url": str(request.url.copy_with(path="/a2a"))})
        data = json.loads(request.content)
        name = request.url.host.split(".")[0]
        requests.append((name, data["method"]))
        if name == "slow":
            slow_started.set()
            if data["method"] == "SendMessage":
                state = "TASK_STATE_WORKING"
            else:
                if mode == "first":
                    await asyncio.sleep(1)
                state = "TASK_STATE_COMPLETED"
            result = {
                "task": {
                    "id": "slow-task",
                    "contextId": "slow-context",
                    "status": {
                        "state": state,
                        "message": {"parts": [{"text": "Longer thoughtful response"}]},
                    },
                }
            }
        else:
            await slow_started.wait()
            result = {"message": {"contextId": "fast-context", "parts": [{"text": "Fast answer"}]}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    config = A2AConfig(
        enabled=True,
        poll_interval=0.05,
        peers={
            name: A2APeer(url=f"https://{name}.example", capabilities=("review",))
            for name in ["slow", "fast"]
        },
    )
    session = Session(provider="fixture", model="test", cwd=tmp_path)
    async with A2AToolset(
        config, home=tmp_path / "home", transport=httpx.MockTransport(respond)
    ) as owner:
        response = await tool(owner, session, "a2a_orchestrate")(
            ToolCall(
                id="coordinate",
                name="a2a_orchestrate",
                arguments={"capability": "review", "message": "Review", "mode": mode},
            )
        )
        assert not response.is_error, response.content
        payload = json.loads(response.content)
        if mode == "best":
            assert payload["results"]["peer"] == "slow"
        else:
            assert payload["results"][-1]["peer"] == "fast"
            assert "may still run" in payload["notice"]
        listed = json.loads(
            (
                await tool(owner, session, "a2a_list")(
                    ToolCall(id="list", name="a2a_list", arguments={})
                )
            ).content
        )
        assert len(listed["conversations"]) == 2
        slow = next(item for item in listed["conversations"] if item["peer"] == "slow")
        if mode == "first":
            assert slow["state"] in {"working", "uncertain"}
        else:
            assert slow["state"] == "completed"
    assert ("slow", "CancelTask") not in requests
    assert requests.count(("slow", "SendMessage")) == 1
