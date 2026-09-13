from __future__ import annotations

import asyncio
import base64
import json
import sys
from email.message import EmailMessage

from harness.cli.channels.extra import EmailTransport, SignalTransport
from harness.cli.channels.runtime import deliver_message
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.core.schemas import MediaAttachment


def test_signal_real_local_rpc_process_and_private_media(tmp_path):
    processes = []
    code = """
import json,sys
message={"jsonrpc":"2.0","method":"receive","params":{"account":"+100","envelope":{"sourceNumber":"+200","sourceDevice":1,"timestamp":123,"dataMessage":{"message":"hello","attachments":[{"id":"file","contentType":"image/png","filename":"image.png"}]}}}}
print(json.dumps(message), flush=True)
print(json.dumps(message), flush=True)
for line in sys.stdin:
    request=json.loads(line)
    result={"data":"aW1hZ2U="} if request["method"]=="getAttachment" else {"timestamp":456,"params":request["params"]}
    print(json.dumps({"jsonrpc":"2.0","id":request["id"],"result":result}), flush=True)
"""

    async def create(*args, **kwargs):
        assert args[1:] == ("-a", "+100", "jsonRpc")
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", code, **kwargs)
        processes.append(process)
        return process

    async def run():
        received = asyncio.Event()

        class SignaledStore(ChannelStore):
            def ingest(self, message):
                result = super().ingest(message)
                received.set()
                return result

        transport = SignalTransport(
            config=ChannelConfig(signal_command=sys.executable, allowed_users=["+200"]),
            token="+100",
            process_factory=create,
        )
        store = SignaledStore(cwd=tmp_path, transport="signal")
        await transport.authenticate()
        task = asyncio.create_task(transport.receive(store))
        try:
            async with asyncio.timeout(3):
                await received.wait()
            assert len(store.status()["inbox"]) == 1
            message = store.claim_message()
            assert message and message.user_id == "+200"
            media = await transport.prepare_media(message)
            assert media[0].data and base64.b64decode(media[0].data) == b"image"
            result = await transport.rpc("send", {"recipient": ["+200"], "message": "response"})
            assert result["params"]["recipient"] == ["+200"]
            await transport.send_media("+200", media[0], "outbound")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert processes[0].returncode is not None
            store.close()
            await transport.close()

    asyncio.run(run())


def mail(*, sender="owner@example.org", reply_id=None, text="approve private", attachment=False):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "bot@example.org"
    message["Subject"] = "Hello"
    if reply_id:
        message["In-Reply-To"] = reply_id
    message.set_content(text)
    if attachment:
        message.add_attachment(b"image", maintype="image", subtype="png", filename="image.png")
    return message.as_bytes()


def test_email_requires_mailbox_possession_before_dispatch_and_approval(tmp_path):
    sent = []

    class SMTP:
        def __init__(self, host, port, **kwargs):
            assert host == "smtp.example.org" and port == 465
            assert kwargs["context"].check_hostname

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def login(self, user, password):
            assert user == "bot@example.org" and password == "private-password"

        def send_message(self, message, *, from_addr, to_addrs):
            assert to_addrs == ["owner@example.org"]
            sent.append(message)

    async def run():
        config = ChannelConfig(
            allowed_users=["owner@example.org", "other@example.org"],
            imap_host="imap.example.org",
            smtp_host="smtp.example.org",
            username="bot@example.org",
        )
        transport = EmailTransport(config=config, token="private-password", smtp_factory=SMTP)
        transport.identity = "bot@example.org"
        store = ChannelStore(cwd=tmp_path, transport="email")
        transport.store = store
        # An attacker can forge From, but the first message only sends a proof
        # challenge to the actual owner's mailbox, never to Reply-To.
        assert transport.parse(mail(), uid="1", store=store) is None
        assert transport.parse(mail(), uid="2", store=store) is None
        assert len(store.status()["outbox"]) == 1
        await deliver_message(transport=transport, store=store)
        challenge = str(sent[0]["Message-ID"])
        assert len(challenge) > 64
        assert challenge not in json.dumps(store.status())
        # A different mailbox cannot reuse the owner's challenge.
        assert (
            transport.parse(
                mail(sender="other@example.org", reply_id=challenge), uid="3", store=store
            )
            is None
        )
        accepted = transport.parse(mail(reply_id=challenge, attachment=True), uid="4", store=store)
        assert accepted and accepted.user_id == "owner@example.org"
        assert accepted.text == "approve private"
        media = await transport.prepare_media(accepted)
        assert media[0].kind == "image"
        # The secret reply capability and captured attachments survive restart.
        store.ingest(accepted)
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="email")
        assert store.claim_message() == accepted
        grant = store.get(f"email_reply:{challenge}")
        grant["expires"] = 0
        store.set(f"email_reply:{challenge}", grant)
        assert transport.parse(mail(reply_id=challenge), uid="5", store=store) is None
        store.close()
        await transport.close()

    asyncio.run(run())


def test_email_imap_cursor_preserved_and_mime_send(tmp_path):
    selected, fetched, sent = [], [], []

    class IMAP:
        def __init__(self, host, port, **kwargs):
            assert host == "imap.example.org" and port == 993
            assert kwargs["ssl_context"].check_hostname

        def login(self, user, password):
            assert password == "secret"

        def select(self, mailbox, readonly):
            selected.append((mailbox, readonly))
            return "OK", [b"1"]

        def response(self, name):
            assert name == "UIDVALIDITY"
            return name, [b"44"]

        def uid(self, command, *args):
            if command == "search":
                return "OK", [b"9"]
            fetched.append(args)
            return "OK", [(b"9 (BODY[])", mail())]

        def logout(self):
            pass

    class SMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def login(self, *args):
            pass

        def send_message(self, message, **kwargs):
            sent.append(message)

    async def run():
        config = ChannelConfig(
            allowed_users=["owner@example.org"],
            imap_host="imap.example.org",
            smtp_host="smtp.example.org",
            username="bot@example.org",
        )
        transport = EmailTransport(
            config=config, token="secret", imap_factory=IMAP, smtp_factory=SMTP
        )
        await transport.authenticate()
        store = ChannelStore(cwd=tmp_path, transport="email")
        await transport.poll_once(store)
        assert store.get("uid") == 9 and selected == [("INBOX", True)]
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="email")
        await transport.poll_once(store)
        assert len(fetched) == 1
        media = MediaAttachment(
            kind="audio",
            mime_type="audio/mpeg",
            data=base64.b64encode(b"audio").decode(),
            name="voice.mp3",
        )
        await transport.send_media("owner@example.org", media, "media-id")
        assert next(sent[0].iter_attachments()).get_payload(decode=True) == b"audio"
        store.close()
        await transport.close()

    asyncio.run(run())
