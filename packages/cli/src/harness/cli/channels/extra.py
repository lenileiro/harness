from __future__ import annotations

import asyncio
import base64
import email.policy
import hashlib
import imaplib
import json
import secrets
import shutil
import smtplib
import ssl
import time
from contextlib import suppress
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any
from uuid import uuid4

from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class SignalTransport(Transport):
    name = "signal"
    limit = 4000

    def __init__(self, **kwargs: Any):
        self.process_factory = kwargs.pop("process_factory", asyncio.create_subprocess_exec)
        super().__init__(**kwargs)
        self.process: Any = None
        self.pending: dict[str, asyncio.Future[Any]] = {}
        self.write_lock = asyncio.Lock()

    async def authenticate(self) -> None:
        command = shutil.which(self.config.signal_command)
        if not command:
            raise ChannelError(
                "Install signal-cli and link/register an account before starting Signal"
            )
        self.identity = self.bot_id = self.token
        self.command = command

    def can_send(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def receive(self, store: ChannelStore) -> None:
        self.process = await self.process_factory(
            self.command,
            "-a",
            self.token,
            "jsonRpc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=30 * 1024 * 1024,
        )
        process = self.process
        try:
            while line := await process.stdout.readline():
                payload = json.loads(line)
                key = str(payload.get("id", ""))
                if key in self.pending:
                    future = self.pending.pop(key)
                    if not future.done():
                        if payload.get("error"):
                            future.set_exception(ChannelError("Signal rejected the RPC request"))
                        else:
                            future.set_result(payload.get("result"))
                elif payload.get("method") == "receive":
                    params = payload.get("params", {})
                    self.accept(self.parse(params.get("result", params)), store)
            raise ChannelError("Signal subprocess disconnected")
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ChannelError("Signal RPC interrupted"))
            self.pending.clear()
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            self.process = None

    def parse(self, payload: dict[str, Any]) -> ChannelMessage | None:
        if payload.get("account") and payload["account"] != self.token:
            return None
        envelope = payload.get("envelope", {})
        data = envelope.get("dataMessage", {})
        user = str(envelope.get("sourceNumber") or envelope.get("sourceUuid") or "")
        text = str(data.get("message") or "")
        if not user or user == self.token or not data or data.get("viewOnce"):
            return None
        group = str(data.get("groupInfo", {}).get("groupId") or "")
        thread = f"group:{group}" if group else user
        mentioned = any(
            str(item.get("number") or item.get("recipientNumber")) == self.token
            for item in data.get("mentions", [])
        )
        attachments = [
            {
                "id": item["id"],
                "mime_type": item.get("contentType") or "application/octet-stream",
                "name": item.get("filename") or "attachment",
            }
            for item in data.get("attachments", [])
            if item.get("id")
        ][:16]
        if not text and not attachments:
            return None
        key = hashlib.sha256(
            json.dumps(
                [user, thread, envelope.get("timestamp"), envelope.get("sourceDevice")]
            ).encode()
        ).hexdigest()
        return ChannelMessage(
            key,
            user,
            thread,
            text or "Please inspect the attached media.",
            thread,
            bool(group),
            mentioned,
            attachments,
        )

    async def rpc(self, method: str, params: dict[str, Any]) -> Any:
        if self.process is None or self.process.returncode is not None:
            raise ChannelError("Signal is not connected")
        key = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[key] = future
        try:
            async with self.write_lock:
                self.process.stdin.write(
                    (
                        json.dumps(
                            {"jsonrpc": "2.0", "id": key, "method": method, "params": params}
                        )
                        + "\n"
                    ).encode()
                )
                await self.process.stdin.drain()
            return await asyncio.wait_for(future, timeout=60)
        finally:
            self.pending.pop(key, None)

    def destination(self, thread_id: str) -> dict[str, Any]:
        return (
            {"groupId": thread_id[6:]}
            if thread_id.startswith("group:")
            else {"recipient": [thread_id]}
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self.rpc("send", {**self.destination(thread_id), "message": text})

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        if attachment.data is None:
            raise ChannelError("Signal media requires inline data")
        await self.rpc(
            "send", {**self.destination(thread_id), "attachments": [attachment.data_uri()]}
        )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result = []
        total = 0
        for item in message.attachments[:16]:
            destination = (
                {"groupId": message.thread_id[6:]}
                if message.group
                else {"recipient": message.user_id}
            )
            response = await self.rpc("getAttachment", {"id": item["id"], **destination})
            data = str(response["data"])
            total += len(base64.b64decode(data, validate=True))
            if total > 20 * 1024 * 1024:
                raise ChannelError("Signal attachments exceed 20 MiB")
            mime = str(item["mime_type"])
            kind = (
                "image"
                if mime.startswith("image/")
                else "audio"
                if mime.startswith("audio/")
                else "file"
            )
            result.append(
                MediaAttachment(kind=kind, mime_type=mime, data=data, name=str(item["name"])[:255])
            )
        return result


class EmailTransport(Transport):
    name = "email"
    limit = 20000

    def __init__(self, **kwargs: Any):
        self.imap_factory = kwargs.pop("imap_factory", imaplib.IMAP4_SSL)
        self.smtp_factory = kwargs.pop("smtp_factory", smtplib.SMTP_SSL)
        super().__init__(**kwargs)
        self.mailbox: Any = None
        self.store: ChannelStore | None = None

    def _connect(self) -> Any:
        mailbox = self.imap_factory(
            self.config.imap_host,
            self.config.imap_port,
            ssl_context=ssl.create_default_context(),
            timeout=30,
        )
        mailbox.login(self.config.username, self.token)
        status, _ = mailbox.select(self.config.mailbox, readonly=True)
        if status != "OK":
            mailbox.logout()
            raise ChannelError("Email mailbox is unavailable")
        return mailbox

    async def authenticate(self) -> None:
        if not self.config.imap_host or not self.config.smtp_host or not self.config.username:
            raise ChannelError("Email requires imap_host, smtp_host, and username")
        self.identity = self.bot_id = self.config.username.lower()
        self.mailbox = await asyncio.to_thread(self._connect)

    def can_send(self) -> bool:
        return self.store is not None

    def parse(self, raw: bytes, *, uid: str, store: ChannelStore) -> ChannelMessage | None:
        message = BytesParser(policy=email.policy.default).parsebytes(raw)
        user = parseaddr(str(message.get("From", "")))[1].lower()
        if user not in self.config.allowed_users or user == self.identity:
            return None
        if message.get("Auto-Submitted", "no") != "no":
            return None
        reply_id = str(message.get("In-Reply-To", "")).strip()
        grant = store.get(f"email_reply:{reply_id}", {})
        if (
            not isinstance(grant, dict)
            or grant.get("user") != user
            or grant.get("expires", 0) < time.time()
        ):
            # From is spoofable. Only send a challenge to the allowlisted mailbox;
            # possession of its random Message-ID authorizes a later reply.
            if store.get(f"email_challenge:{user}", 0) < time.time():
                store.queue(
                    source=f"email-challenge:{uid}",
                    user_id=user,
                    thread_id=user,
                    text="Reply to this message to confirm access to this mailbox and continue with Harness. Requests and approvals are accepted only as replies to a Harness message.",
                    limit=self.limit,
                )
                store.set(f"email_challenge:{user}", time.time() + 3600)
            return None
        body = message.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body is not None else ""
        if body is not None and body.get_content_type() == "text/html":

            class TextOnly(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text: list[str] = []
                    self.hidden = 0

                def handle_starttag(self, tag, attrs):
                    if tag in {"script", "style"}:
                        self.hidden += 1

                def handle_endtag(self, tag):
                    if tag in {"script", "style"}:
                        self.hidden = max(0, self.hidden - 1)

                def handle_data(self, data):
                    if not self.hidden:
                        self.text.append(data)

            parser = TextOnly()
            parser.feed(str(text))
            text = " ".join(parser.text)
        attachments = []
        total = 0
        for part in message.iter_attachments():
            data = part.get_payload(decode=True) or b""
            if not isinstance(data, bytes):
                raise ChannelError("Invalid MIME attachment payload")
            total += len(data)
            if total > 20 * 1024 * 1024 or len(attachments) >= 16:
                raise ChannelError("Email attachments exceed the supported size/count")
            mime = part.get_content_type()
            kind = (
                "image"
                if mime.startswith("image/")
                else "audio"
                if mime.startswith("audio/")
                else "file"
            )
            attachments.append(
                MediaAttachment(
                    kind=kind,
                    mime_type=mime,
                    data=base64.b64encode(data).decode(),
                    name=part.get_filename(),
                ).model_dump(mode="json")
            )
        message_key = hashlib.sha256(
            json.dumps(
                [user, str(message.get("Message-ID") or hashlib.sha256(raw).hexdigest())]
            ).encode()
        ).hexdigest()
        return ChannelMessage(
            message_key,
            user,
            user,
            str(text).strip() or "Please inspect the attached media.",
            user,
            attachments=attachments,
        )

    async def poll_once(self, store: ChannelStore) -> None:
        if self.mailbox is None:
            self.mailbox = await asyncio.to_thread(self._connect)
        self.store = store
        _, validity = await asyncio.to_thread(self.mailbox.response, "UIDVALIDITY")
        current = str(validity[0]) if validity else "unknown"
        if current != store.get("uidvalidity"):
            store.set("uidvalidity", current)
            store.set("uid", 0)
        lower = int(store.get("uid", 0)) + 1
        status, values = await asyncio.to_thread(self.mailbox.uid, "search", None, f"UID {lower}:*")
        if status != "OK":
            raise ChannelError("Email UID search failed")
        for value in values[0].split() if values else []:
            uid = int(value)
            if uid < lower:
                continue
            status, parts = await asyncio.to_thread(
                self.mailbox.uid, "fetch", str(uid), "(BODY.PEEK[])"
            )
            if status != "OK":
                raise ChannelError("Email fetch failed")
            raw = next((part[1] for part in parts if isinstance(part, tuple)), b"")
            if len(raw) > 30 * 1024 * 1024:
                raise ChannelError("Email message exceeds 30 MiB")
            self.accept(self.parse(raw, uid=f"{current}:{uid}", store=store), store)
            store.set("uid", uid)

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        try:
            while True:
                await self.poll_once(store)
                await asyncio.sleep(15)
        finally:
            if self.mailbox is not None:
                with suppress(Exception):
                    await asyncio.to_thread(self.mailbox.logout)
                self.mailbox = None

    def _send_mail(self, message: EmailMessage, recipient: str) -> None:
        with self.smtp_factory(
            self.config.smtp_host,
            self.config.smtp_port,
            context=ssl.create_default_context(),
            timeout=30,
        ) as smtp:
            smtp.login(self.config.username, self.token)
            smtp.send_message(message, from_addr=self.config.username, to_addrs=[recipient])

    async def _post(
        self, thread_id: str, text: str, delivery_id: str, attachment: MediaAttachment | None = None
    ) -> None:
        if self.store is None or thread_id not in self.config.allowed_users:
            raise ChannelError("Email destination is not authorized")
        key = f"email_outbound:{delivery_id}"
        message_id = self.store.get(key)
        if not message_id:
            message_id = f"<harness-{secrets.token_hex(32)}@localhost>"
            self.store.set(key, message_id)
            self.store.set(
                f"email_reply:{message_id}", {"user": thread_id, "expires": time.time() + 86400}
            )
        message = EmailMessage()
        message["From"], message["To"] = self.config.username, thread_id
        message["Subject"] = "Harness"
        message["Message-ID"] = message_id
        message["Auto-Submitted"] = "auto-replied"
        message.set_content(text or "Attached media from Harness.")
        if attachment:
            if attachment.data is None:
                raise ChannelError("Email media requires inline data")
            maintype, subtype = attachment.mime_type.split("/", 1)
            message.add_attachment(
                base64.b64decode(attachment.data, validate=True),
                maintype=maintype,
                subtype=subtype,
                filename=attachment.name or "attachment",
            )
        await asyncio.to_thread(self._send_mail, message, thread_id)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self._post(thread_id, text, delivery_id)

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        await self._post(thread_id, "", delivery_id, attachment)

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        return [MediaAttachment.model_validate(item) for item in message.attachments]

    async def close(self) -> None:
        if self.mailbox is not None:
            with suppress(Exception):
                await asyncio.to_thread(self.mailbox.logout)
            self.mailbox = None
        await super().close()
