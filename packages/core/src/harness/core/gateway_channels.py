"""Authenticated channel configuration and durable inbox/outbox state."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHANNEL_LIMITS = {
    "telegram": 4096,
    "discord": 2000,
    "slack": 4000,
    "signal": 4000,
    "email": 20000,
    "matrix": 4000,
    "google_chat": 4000,
    "feishu": 4000,
    "lark": 4000,
    "teams": 4000,
    "dingtalk": 4000,
    "mattermost": 4000,
    "ntfy": 1000,
    "irc": 100,
    "line": 5000,
    "wecom": 500,
    "sms": 1000,
    "simplex": 4000,
    "buzz": 4000,
    "raft": 4000,
    "photon": 4000,
    "yuanbao": 4000,
    "whatsapp_cloud": 4096,
    "bluebubbles": 4000,
    "msgraph_webhook": 4000,
    "weixin": 2000,
    "qqbot": 4000,
}


@dataclass(frozen=True)
class ChannelConfig:
    token_env: str = ""
    app_token_env: str = ""
    allowed_users: list[str] = field(default_factory=list)
    allowed_channels: list[str] = field(default_factory=list)
    allow_groups: bool = False
    require_mention: bool = True
    operator_users: list[str] = field(default_factory=list)
    profile: str = ""
    signal_command: str = "signal-cli"
    imap_host: str = ""
    smtp_host: str = ""
    username: str = ""
    imap_port: int = 993
    smtp_port: int = 465
    mailbox: str = "INBOX"
    homeserver: str = ""
    matrix_e2ee: bool = False
    matrix_device_id: str = ""
    matrix_store_path: str = ""
    matrix_pickle_key_env: str = ""
    matrix_trusted_devices: dict[str, dict[str, str]] = field(default_factory=dict)
    matrix_unverified_policy: str = "reject"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8766
    webhook_path: str = "/events"
    audience: str = ""
    app_id: str = ""
    signing_secret_env: str = ""
    tenant_id: str = ""
    account_file: str = ""
    receive_mode: str = "websocket"
    wecom_mode: str = "internal_app"
    command: str = ""
    topic: str = ""
    reply_topic: str = ""
    phone_number_id: str = ""
    waba_id: str = ""
    api_version: str = "v23.0"
    webhook_url: str = ""
    subscription_id: str = ""
    accepted_resources: list[str] = field(default_factory=list)
    allowed_targets: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelConfig:
        fields = cls.__dataclass_fields__
        if unknown := set(data) - set(fields):
            raise ValueError(f"Unknown channel configuration fields: {', '.join(sorted(unknown))}")
        for name in (
            "allowed_users",
            "allowed_channels",
            "operator_users",
            "accepted_resources",
            "allowed_targets",
        ):
            value = data.get(name, [])
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"channels.{name} must be a list of nonempty identity strings")
        for name in ("allow_groups", "require_mention", "matrix_e2ee"):
            if name in data and not isinstance(data[name], bool):
                raise ValueError(f"channels.{name} must be boolean")
        for name in (
            "token_env",
            "app_token_env",
            "profile",
            "signal_command",
            "imap_host",
            "smtp_host",
            "username",
            "mailbox",
            "homeserver",
            "matrix_device_id",
            "matrix_store_path",
            "matrix_pickle_key_env",
            "matrix_unverified_policy",
            "listen_host",
            "webhook_path",
            "audience",
            "app_id",
            "signing_secret_env",
            "tenant_id",
            "account_file",
            "receive_mode",
            "wecom_mode",
            "command",
            "topic",
            "reply_topic",
            "phone_number_id",
            "waba_id",
            "api_version",
            "webhook_url",
            "subscription_id",
        ):
            if name in data and not isinstance(data[name], str):
                raise ValueError(f"channels.{name} must be a string")
        for name in ("imap_port", "smtp_port", "listen_port"):
            if name in data and (type(data[name]) is not int or not 1 <= data[name] <= 65535):
                raise ValueError(f"channels.{name} must be a valid port")
        devices = data.get("matrix_trusted_devices", {})
        if data.get("wecom_mode", "internal_app") not in {"internal_app", "bot"}:
            raise ValueError("channels.wecom_mode must be internal_app or bot")
        if not isinstance(devices, dict) or any(
            not isinstance(user, str)
            or not user.startswith("@")
            or not isinstance(keys, dict)
            or any(
                not isinstance(device, str) or not device or not isinstance(key, str) or not key
                for device, key in keys.items()
            )
            for user, keys in devices.items()
        ):
            raise ValueError(
                "channels.matrix_trusted_devices must map user IDs to device IDs and exact Ed25519 fingerprints"
            )
        if data.get("matrix_unverified_policy", "reject") != "reject":
            raise ValueError(
                "channels.matrix_unverified_policy must be reject; blanket trust is unavailable"
            )
        if data.get("matrix_e2ee") and any(
            not data.get(name)
            for name in ("matrix_device_id", "matrix_store_path", "matrix_pickle_key_env")
        ):
            raise ValueError(
                "Matrix E2EE requires matrix_device_id, matrix_store_path and matrix_pickle_key_env"
            )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def permits(self, *, user_id: str, channel_id: str, group: bool, mentioned: bool) -> bool:
        return (
            user_id in self.allowed_users
            and (not self.allowed_channels or channel_id in self.allowed_channels)
            and (not group or (self.allow_groups and (not self.require_mention or mentioned)))
        )


@dataclass(frozen=True)
class ChannelMessage:
    id: str
    user_id: str
    thread_id: str
    text: str
    channel_id: str
    group: bool = False
    mentioned: bool = False
    attachments: list[dict[str, Any]] = field(default_factory=list)
    question_capture_id: str = ""


def split_channel_text(text: str, limit: int) -> list[str]:
    """Bound UTF-16 units too, preserving every character, including astral emoji."""
    chunks: list[str] = []
    current: list[str] = []
    units = 0
    for character in text or "(empty response)":
        size = 2 if ord(character) > 0xFFFF else 1
        if current and units + size > limit:
            chunks.append("".join(current))
            current, units = [], 0
        current.append(character)
        units += size
    if current:
        chunks.append("".join(current))
    return chunks


class ChannelStore:
    def __init__(self, *, cwd: Path, transport: str):
        if transport not in CHANNEL_LIMITS:
            raise ValueError("Unsupported channel")
        self.path = cwd / ".harness" / "channels" / f"{transport}.db"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path, timeout=10)
        os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inbox (
                id TEXT PRIMARY KEY, message TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                created REAL NOT NULL, error TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL, text TEXT NOT NULL, part INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '');
        """)

        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if "attachment" not in {
                row["name"] for row in self.db.execute("PRAGMA table_info(outbox)")
            }:
                self.db.execute("ALTER TABLE outbox ADD COLUMN attachment TEXT NOT NULL DEFAULT ''")
            if "interaction" not in {
                row["name"] for row in self.db.execute("PRAGMA table_info(outbox)")
            }:
                self.db.execute(
                    "ALTER TABLE outbox ADD COLUMN interaction TEXT NOT NULL DEFAULT ''"
                )
        from harness.core.channel_interactions import initialize

        initialize(self)
        from harness.core.question_interactions import initialize as initialize_questions

        initialize_questions(self)

    def close(self) -> None:
        self.db.close()

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def bind_identity(self, identity: str) -> None:
        previous = self.get("identity")
        if previous is not None and previous != identity:
            raise ValueError(
                "This channel state belongs to another bot; use a separate profile/workspace"
            )
        self.set("identity", identity)

    def ingest(self, message: ChannelMessage) -> bool:
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO inbox (id,message,created) VALUES (?,?,?)",
                (message.id, json.dumps(asdict(message)), time.time()),
            )
        return bool(cursor.rowcount)

    def recover(self) -> None:
        # Model/tool work and a network send may have happened before a crash.
        # Neither is safe to repeat automatically after an unfinished claim.
        with self.db:
            self.db.execute(
                "UPDATE inbox SET status='uncertain',error='Interrupted dispatch; inspect session before retry' WHERE status='processing'"
            )
            self.db.execute(
                "UPDATE outbox SET status='uncertain',error='Interrupted send; inspect destination before retry' WHERE status='sending'"
            )

    def claim_message(self) -> ChannelMessage | None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM inbox WHERE status='pending' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            self.db.execute("UPDATE inbox SET status='processing' WHERE id=?", (row["id"],))
        return ChannelMessage(**json.loads(row["message"]))

    def _queue(
        self,
        *,
        source: str,
        user_id: str,
        thread_id: str,
        text: str,
        limit: int,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        for part, chunk in enumerate(split_channel_text(text, limit)):
            key = hashlib.sha256(json.dumps([source, part]).encode()).hexdigest()
            self.db.execute(
                "INSERT OR IGNORE INTO outbox (id,source,user_id,thread_id,text,part) VALUES (?,?,?,?,?,?)",
                (key, source, user_id, thread_id, chunk, part),
            )
        offset = len(split_channel_text(text, limit))
        for index, attachment in enumerate((attachments or [])[:16], start=offset):
            key = hashlib.sha256(json.dumps([source, index]).encode()).hexdigest()
            self.db.execute(
                "INSERT OR IGNORE INTO outbox (id,source,user_id,thread_id,text,part,attachment) VALUES (?,?,?,?,?,?,?)",
                (key, source, user_id, thread_id, "", index, json.dumps(attachment)),
            )

    def complete(
        self,
        message: ChannelMessage,
        text: str,
        *,
        limit: int,
        attachments: list[dict[str, Any]] | None = None,
        approval_cards: list[dict[str, Any]] | None = None,
        question_card: dict[str, Any] | None = None,
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE question_interactions SET mode='used' WHERE mode='capturing' AND capture_message=? AND user_id=? AND thread_id=?",
                (message.id, message.user_id, message.thread_id),
            )
            self._queue(
                source=f"reply:{message.id}",
                user_id=message.user_id,
                thread_id=message.thread_id,
                text=text,
                limit=limit,
                attachments=attachments,
            )
            if approval_cards:
                from harness.core.channel_interactions import queue_cards

                queue_cards(
                    self,
                    message,
                    approval_cards,
                    start=len(split_channel_text(text, limit)) + len((attachments or [])[:16]),
                )
            if question_card:
                from harness.core.question_interactions import queue_card

                start = self.db.execute(
                    "SELECT COALESCE(MAX(part),-1)+1 FROM outbox WHERE source=?",
                    ("reply:" + message.id,),
                ).fetchone()[0]
                queue_card(self, message, question_card, start=start)
            self.db.execute("UPDATE inbox SET status='completed' WHERE id=?", (message.id,))

    def queue(
        self,
        *,
        source: str,
        user_id: str,
        thread_id: str,
        text: str,
        limit: int,
        approval_cards: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.db:
            self._queue(source=source, user_id=user_id, thread_id=thread_id, text=text, limit=limit)
            if approval_cards:
                from harness.core.channel_interactions import queue_cards

                context = self.db.execute(
                    """SELECT message FROM inbox WHERE json_extract(message,'$.user_id')=?
                    AND json_extract(message,'$.thread_id')=? ORDER BY created DESC LIMIT 1""",
                    (user_id, thread_id),
                ).fetchone()
                if context:
                    queue_cards(
                        self,
                        ChannelMessage(**json.loads(context["message"])),
                        approval_cards,
                        start=len(split_channel_text(text, limit)),
                        source=source,
                    )

    def fail_message(self, message: ChannelMessage) -> None:
        with self.db:
            self.db.execute(
                "UPDATE inbox SET status='uncertain',error='Dispatch interrupted; inspect session before retry' WHERE id=?",
                (message.id,),
            )

    def claim_delivery(self) -> dict[str, Any] | None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                """SELECT * FROM outbox AS current WHERE status='pending' AND next_attempt<=?
                AND NOT EXISTS (SELECT 1 FROM outbox AS earlier WHERE earlier.source=current.source
                    AND earlier.part<current.part AND earlier.status!='sent')
                ORDER BY rowid LIMIT 1""",
                (time.time(),),
            ).fetchone()
            if not row:
                return None
            self.db.execute(
                "UPDATE outbox SET status='sending',attempts=attempts+1 WHERE id=?", (row["id"],)
            )
        return dict(row)

    def delivery_result(
        self, key: str, *, status: str, retry_after: float = 0, error: str = ""
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE outbox SET status=?,next_attempt=?,error=? WHERE id=?",
                (status, time.time() + retry_after, error, key),
            )

    def status(self, *, include_private: bool = False) -> dict[str, Any]:
        return {
            "identity": self.get("identity"),
            "connection": self.get("connection", "stopped"),
            "connection_error": self.get("connection_error", ""),
            "subscription_lifecycle": self.get("subscription_lifecycle"),
            "matrix_encryption": self.get("matrix_encryption"),
            **({"local_responses": self.get("local_responses", [])} if include_private else {}),
            "inbox": [
                dict(row)
                for row in self.db.execute(
                    "SELECT id,status,error FROM inbox ORDER BY created DESC LIMIT 30"
                )
            ],
            "outbox": [
                dict(row)
                for row in self.db.execute(
                    "SELECT id,source,part,status,attempts,error FROM outbox ORDER BY rowid DESC LIMIT 30"
                )
            ],
        }

    def retry(self, key: str, *, inbound: bool = False) -> bool:
        table = "inbox" if inbound else "outbox"
        with self.db:
            cursor = self.db.execute(
                f"UPDATE {table} SET status='pending',error='' WHERE id=? AND status IN ('uncertain','failed')",
                (key,),
            )
        return bool(cursor.rowcount)
