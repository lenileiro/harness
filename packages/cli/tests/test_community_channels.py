import asyncio
import base64
import json
import ssl
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channels.irc import IRCTransport, parse_line
from harness.cli.channels.mattermost import MattermostTransport
from harness.cli.channels.ntfy import NtfyTransport, sign_ntfy_message
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import ChannelError
from harness.core.gateway_channels import ChannelConfig, ChannelStore


async def test_mattermost_websocket_and_rest_recovery_preserve_thread_and_dedup(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="mattermost")
    store.set("since:channel", 100)
    old = {
        "id": "post-1",
        "user_id": "owner",
        "channel_id": "channel",
        "create_at": 200,
        "message": "@harness start",
        "root_id": "root",
    }
    new = {**old, "id": "post-2", "create_at": 300, "message": "@harness approve pending"}
    requests = []

    async def server(socket):
        frame = json.loads(await socket.recv())
        assert frame == {
            "seq": 1,
            "action": "authentication_challenge",
            "data": {"token": "secret"},
        }
        await socket.send(json.dumps({"seq_reply": 1, "status": "OK"}))
        for post in (old, new, new):
            await socket.send(json.dumps({"event": "posted", "data": {"post": json.dumps(post)}}))
        await socket.close()

    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path.endswith("users/me"):
            return httpx.Response(200, json={"id": "bot", "username": "harness"})
        if request.url.path.endswith("channels/channel"):
            return httpx.Response(200, json={"id": "channel", "type": "O"})
        if request.method == "GET":
            return httpx.Response(200, json={"order": ["post-1"], "posts": {"post-1": old}})
        return httpx.Response(201, json={"id": "sent"})

    async with serve(server, "127.0.0.1", 0) as local:

        def socket_connect(url, **kwargs):
            assert url == "wss://mattermost.example/api/v4/websocket"
            return connect(f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kwargs)

        transport = MattermostTransport(
            config=ChannelConfig(
                homeserver="https://mattermost.example",
                allowed_users=["owner"],
                allowed_channels=["channel"],
                allow_groups=True,
            ),
            token="secret",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            socket_connect=socket_connect,
        )
        await transport.authenticate()
        await transport.receive(store)
    assert len(store.status()["inbox"]) == 2 and store.get("since:channel") == 300
    received = []

    async def receiver(**kwargs):
        received.append(kwargs)
        return {"reply": {"text": "Received"}}

    while await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver):
        pass
    assert {entry["thread_id"] for entry in received} == {'["channel","root"]'}
    assert received[1]["message"] == "approve pending"
    assert await deliver_message(transport=transport, store=store)
    body = json.loads(requests[-1].content)
    assert body["root_id"] == "root" and body["channel_id"] == "channel"
    assert body["pending_post_id"]
    store.close()
    await transport.close()


async def test_mattermost_history_gap_does_not_advance_cursor_or_run_partial_history(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="mattermost")
    store.set("since:channel", 1)
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        posts = [{"id": f"{count}-{index}", "create_at": 1000 - index} for index in range(100)]
        return httpx.Response(
            200,
            json={
                "order": [post["id"] for post in posts],
                "posts": {post["id"]: post for post in posts},
            },
        )

    transport = MattermostTransport(
        config=ChannelConfig(homeserver="https://mattermost.example"),
        token="secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport.channels = {"channel": {"type": "D"}}
    with pytest.raises(ChannelError, match="recovery bound"):
        await transport.catch_up(store)
    assert count == 100 and store.get("since:channel") == 1 and store.claim_message() is None
    store.close()
    await transport.close()


def ntfy_config():
    return ChannelConfig(
        homeserver="https://ntfy.example",
        topic="input",
        reply_topic="output",
        username="owner",
        allowed_users=["owner"],
    )


async def test_ntfy_signed_owner_nonce_replay_freshness_and_cursor(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="ntfy")
    secret = "s" * 32
    valid = sign_ntfy_message(
        text="approve pending", topic="input", owner="owner", secret=secret, message_id="nonce"
    )
    expired = sign_ntfy_message(
        text="old", topic="input", owner="owner", secret=secret, timestamp=int(time.time()) - 90000
    )
    other = sign_ntfy_message(text="private", topic="input", owner="stranger", secret=secret)
    events = [
        {"event": "message", "topic": "input", "id": str(index), "message": body}
        for index, body in enumerate(
            (valid, valid, valid.replace("pending", "forged"), expired, other, "unsigned text")
        )
    ]
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer access-secret"
        if request.method == "POST":
            return httpx.Response(200, json={"id": "published"})
        return httpx.Response(200, text="".join(json.dumps(event) + "\n" for event in events))

    transport = NtfyTransport(
        config=ntfy_config(),
        token="access-secret",
        app_token=secret,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    await transport.poll_once(store)  # Initial cache establishes baseline only.
    assert store.claim_message() is None and store.get("since") == "5"
    await transport.poll_once(store)
    assert requests[-1].url.params["since"] == "5"
    message = store.claim_message()
    assert message and message.user_id == "owner" and message.text == "approve pending"
    assert store.claim_message() is None  # Duplicate nonce across platform events rejected.
    await transport.send("input", "Result", "delivery")
    assert json.loads(requests[-1].content) == {"topic": "output", "message": "Result"}
    assert "s" * 32 not in requests[-1].content.decode()
    store.close()
    await transport.close()


async def test_ntfy_truncated_cache_preserves_cursor_and_rejects_unknown_destination(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="ntfy")
    store.set("since", "last")
    transport = NtfyTransport(
        config=ntfy_config(),
        token="secret",
        app_token="s" * 32,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, headers={"X-Messages-Truncated": "1"}, text="")
            )
        ),
    )
    await transport.authenticate()
    with pytest.raises(ChannelError, match="truncated"):
        await transport.poll_once(store)
    assert store.get("since") == "last"
    with pytest.raises(ChannelError, match="outside"):
        await transport.send("attacker-topic", "private", "id")
    with pytest.raises(ChannelError, match="4096"):
        sign_ntfy_message(text="x" * 4096, topic="input", owner="owner", secret="s" * 32)
    store.close()
    await transport.close()


def tls_contexts(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certfile, keyfile = tmp_path / "cert.pem", tmp_path / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certfile, keyfile)
    client = ssl.create_default_context(cafile=str(certfile))
    return server, client


async def test_irc_actual_tls_sasl_account_identity_dedup_and_reply(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="irc")
    server_ssl, client_ssl = tls_contexts(tmp_path)
    replies = []
    server_finished = asyncio.Event()

    async def server(reader, writer):
        async def receive():
            return (await reader.readline()).decode().rstrip("\r\n")

        async def send(line):
            writer.write((line + "\r\n").encode())
            await writer.drain()

        try:
            assert await receive() == "CAP LS 302"
            assert await receive() == "NICK harness"
            assert (await receive()).startswith("USER harness")
            await send(":server CAP * LS :sasl=PLAIN account-tag server-time message-tags")
            assert (await receive()).startswith("CAP REQ :")
            await send(":server CAP harness ACK :sasl account-tag server-time message-tags")
            assert await receive() == "AUTHENTICATE PLAIN"
            await send("AUTHENTICATE +")
            encoded = (await receive()).split()[1]
            assert base64.b64decode(encoded) == b"botaccount\0botaccount\0secret"
            await send(":server 903 harness :Success")
            assert await receive() == "CAP END"
            await send(":server 001 harness :Welcome")
            assert await receive() == "JOIN #room"
            await send(":harness!bot@server JOIN #room")
            for line in (
                "@account=owner;msgid=one :nick!u@h PRIVMSG #room :harness: approve pending",
                "@account=owner;msgid=one :nick!u@h PRIVMSG #room :harness: approve pending",
                "@account=stranger;msgid=two :nick!u@h PRIVMSG #room :harness: approve pending",
                "@msgid=three :nick!u@h PRIVMSG #room :harness: approve pending",
                "@account=owner;msgid=four :nick!u@h PRIVMSG harness :private command",
                "PING :keepalive",
            ):
                await send(line)
            replies.append(await receive())
            replies.append(await receive())
        finally:
            writer.close()
            await writer.wait_closed()
            server_finished.set()

    local = await asyncio.start_server(server, "127.0.0.1", 0, ssl=server_ssl)

    async def stream_connect(host, port, **kwargs):
        assert host == "irc.example" and port == 6697
        assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
        assert kwargs["server_hostname"] == "irc.example"
        return await asyncio.open_connection(
            "127.0.0.1",
            local.sockets[0].getsockname()[1],
            ssl=client_ssl,
            server_hostname="localhost",
            limit=kwargs["limit"],
        )

    transport = IRCTransport(
        config=ChannelConfig(
            homeserver="ircs://irc.example",
            username="botaccount",
            app_id="harness",
            allowed_users=["irc.example:6697:owner"],
            allowed_channels=["#room"],
            allow_groups=True,
        ),
        token="secret",
        stream_connect=stream_connect,
    )
    try:
        await transport.authenticate()
        task = asyncio.create_task(transport.receive(store))
        for _ in range(50):
            if store.status()["inbox"]:
                break
            await asyncio.sleep(0.01)
        message = store.claim_message()
        assert (
            message
            and message.user_id == "irc.example:6697:owner"
            and message.text == "approve pending"
        )
        assert store.claim_message() is None
        await transport.send("#room", "Result\n/QUIT\x01", "id")
        await asyncio.wait_for(server_finished.wait(), timeout=5)
        with pytest.raises(ChannelError, match="closed"):
            await task
        assert "PONG :keepalive" in replies
        outgoing = next(reply for reply in replies if reply.startswith("PRIVMSG"))
        assert outgoing == "PRIVMSG #room :Result↵/QUIT "
        with pytest.raises(ChannelError, match="outside"):
            await transport.send("nick", "private", "id")
    finally:
        await transport.close()
        local.close()
        await local.wait_closed()
        store.close()


def test_irc_tag_escaping_and_client_tag_cannot_supply_account():
    tags, prefix, command, params = parse_line(
        r"@+account=owner;label=a\sb\:c :nick!user@host PRIVMSG #room :message"
    )
    assert tags == {"+account": "owner", "label": "a b;c"} and "account" not in tags
    assert prefix == "nick!user@host" and command == "PRIVMSG" and params == ["#room", "message"]
