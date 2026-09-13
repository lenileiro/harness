import asyncio
import hashlib
import hmac
import json

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.yuanbao import API_URL, SERVICE, WS_URL, YuanbaoTransport
from harness.cli.channels.yuanbao_proto import Frame, binary, field, number, parse, string
from harness.core.gateway_channels import ChannelConfig, ChannelStore


def inbound(*, sender="owner", identifier="event", group=False, mention=False):
    body = (
        field(1, "Group.CallbackAfterSendMsg" if group else "C2C.CallbackAfterSendMsg")
        + field(2, sender)
        + field(3, "bot")
        + field(12, identifier)
        + field(18, 1 if group else 2)
        + field(13, field(1, "TIMTextElem") + field(2, field(1, "approve request")))
    )
    if group:
        body += field(6, "room")
    if mention:
        custom = json.dumps({"elem_type": 1002, "user_id": "bot"})
        body += field(13, field(1, "TIMCustomElem") + field(2, field(4, custom)))
    return body


async def test_yuanbao_sign_bind_real_binary_socket_ack_after_store_and_send(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="yuanbao")
    acks = []
    sent = []
    ready = asyncio.Event()
    token_requests = []

    def http(request):
        assert str(request.url) == API_URL
        body = json.loads(request.content)
        expected = hmac.new(
            b"app-secret",
            (body["nonce"] + body["timestamp"] + "app-key" + "app-secret").encode(),
            hashlib.sha256,
        ).hexdigest()
        assert body["signature"] == expected and body["app_key"] == "app-key"
        assert request.headers["X-Instance-Id"] == "16"
        token_requests.append(body)
        return httpx.Response(
            200, json={"code": 0, "data": {"token": "signed-secret", "bot_id": "bot"}}
        )

    async def server(socket):
        bind = Frame.decode(await socket.recv())
        assert bind.command == "auth-bind" and bind.module == "conn_access"
        data = parse(bind.data)
        assert string(data, 1) == "ybBot" and number(data, 6) == 1
        auth = parse(binary(data, 2))
        assert string(auth, 1) == "bot" and string(auth, 3) == "signed-secret"
        await socket.send(
            Frame(1, bind.command, bind.identifier, bind.module, field(3, "connection")).encode()
        )
        events = [
            inbound(),
            inbound(),
            inbound(sender="stranger", identifier="other"),
            inbound(identifier="group", group=True),
            inbound(identifier="mentioned", group=True, mention=True),
        ]
        for index, event in enumerate(events):
            # Exercise Tencent's optional PushMsg wrapper too.
            if index == 4:
                event = field(1, "message") + field(2, SERVICE) + field(4, event)
            await socket.send(
                Frame(2, "message", str(index), SERVICE, event, need_ack=True).encode()
            )
        async for raw in socket:
            frame = Frame.decode(raw)
            if frame.kind == 3:
                acks.append(frame.identifier)
                if frame.identifier == "0":
                    assert store.status()["inbox"][0]["status"] == "pending"
                if len(acks) == 5:
                    ready.set()
            elif frame.command == "ping":
                await socket.send(
                    Frame(1, "ping", frame.identifier, "conn_access", field(1, 60)).encode()
                )
            elif frame.command == "send_c2c_message":
                data = parse(frame.data)
                assert string(data, 2) == "owner" and string(data, 3) == "bot"
                message = parse(binary(data, 5))
                assert string(message, 1) == "TIMTextElem"
                sent.append(string(parse(binary(message, 2)), 1))
                await socket.send(
                    Frame(1, frame.command, frame.identifier, SERVICE, field(1, 0)).encode()
                )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(http)) as client,
        serve(server, "127.0.0.1", 0) as local,
    ):

        def socket_factory(url, **kwargs):
            assert url == WS_URL
            return connect(f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kwargs)

        transport = YuanbaoTransport(
            config=ChannelConfig(app_id="app-key", allowed_users=["owner"], allow_groups=True),
            token="app-secret",
            client=client,
            socket_connect=socket_factory,
        )
        await transport.authenticate()
        store.bind_identity(transport.identity)
        task = asyncio.create_task(transport.receive(store))
        await asyncio.wait_for(ready.wait(), 3)
        first, second = store.claim_message(), store.claim_message()
        assert first and second and first.user_id == second.user_id == "owner"
        assert first.thread_id == '["bot","dm","owner",""]'
        assert second.thread_id == '["bot","group","room",""]'
        assert store.claim_message() is None
        await transport.send(first.thread_id, "reply 😀", "delivery")
        assert sent == ["reply 😀"] and len(token_requests) == 2
        with pytest.raises(ChannelError, match="another bot"):
            await transport.send('["other","dm","owner",""]', "secret", "bad")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.close()
    store.close()
    reopened = ChannelStore(cwd=tmp_path, transport="yuanbao")
    transport.ingest(inbound(), reopened)
    assert len(reopened.status()["inbox"]) == 2
    reopened.close()


async def test_yuanbao_bot_changes_rate_limit_and_bad_auth_fail_closed():
    values = [
        httpx.Response(200, json={"code": 0, "data": {"token": "secret", "bot_id": "one"}}),
        httpx.Response(200, json={"code": 0, "data": {"token": "secret", "bot_id": "two"}}),
        httpx.Response(429, headers={"Retry-After": "7"}),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: values.pop(0))) as client:
        transport = YuanbaoTransport(
            config=ChannelConfig(app_id="app"), token="secret", client=client
        )
        await transport.authenticate()
        with pytest.raises(ChannelError, match="identity changed"):
            await transport.sign()
        with pytest.raises(RateLimited) as error:
            await transport.sign()
        assert error.value.delay == 7

        class FakeSocket:
            async def send(self, data):
                self.bind = Frame.decode(data)

            async def recv(self):
                return Frame(
                    1,
                    "auth-bind",
                    self.bind.identifier,
                    "conn_access",
                    field(1, 41102) + field(3, "connection"),
                ).encode()

        with pytest.raises(ChannelError, match="authentication rejected"):
            await transport.bind(FakeSocket())
        await transport.close()


@pytest.mark.parametrize("data", [b"\x80", b"\xff" * 11, b"\x00", b"\x0a\xff\x7f", b"\x0b"])
def test_yuanbao_rejects_malformed_protobuf(data):
    with pytest.raises(ValueError):
        parse(data)
