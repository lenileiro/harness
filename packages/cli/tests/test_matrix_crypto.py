from __future__ import annotations

import base64
import json
from dataclasses import replace
from urllib.parse import unquote

import httpx
import pytest

pytest.importorskip("nio", reason="Install the CLI matrix extra for real crypto tests")
pytest.importorskip("vodozemac", reason="Matrix E2EE needs vodozemac native wheels")

from harness.cli.channels.matrix import MatrixTransport
from harness.cli.channels.transports import ChannelError
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.core.schemas import MediaAttachment

ROOM = "!room:matrix.invalid"
ALICE = "@alice:matrix.invalid"
BOB = "@bob:matrix.invalid"


class Homeserver:
    """Offline signed-key exchange using real SDK-generated Olm/Megolm keys."""

    def __init__(self):
        self.keys = {}
        self.one_time = {}
        self.to_device = {ALICE: [], BOB: []}
        self.timeline = {ALICE: [], BOB: []}
        self.requests = []
        self.counter = 0
        self.media = {}
        self.encrypted = True

    def request(self, request):
        self.requests.append(request)
        token = request.headers["authorization"].removeprefix("Bearer ")
        user = ALICE if token == "alice-token" else BOB
        device = "ALICE" if user == ALICE else "BOB"
        path = unquote(request.url.path)
        if path.endswith("/account/whoami"):
            return httpx.Response(200, json={"user_id": user, "device_id": device})
        body = json.loads(request.content) if request.content and "upload" not in path else {}
        if path.endswith("/keys/upload"):
            body = json.loads(request.content)
            if body.get("device_keys"):
                self.keys.setdefault(user, {})[device] = body["device_keys"]
            self.one_time.setdefault((user, device), {}).update(body.get("one_time_keys", {}))
            return httpx.Response(
                200,
                json={
                    "one_time_key_counts": {"signed_curve25519": len(self.one_time[(user, device)])}
                },
            )
        if path.endswith("/keys/query"):
            return httpx.Response(
                200,
                json={
                    "device_keys": {name: self.keys.get(name, {}) for name in body["device_keys"]},
                    "failures": {},
                },
            )
        if path.endswith("/keys/claim"):
            result = {}
            for name, devices in body["one_time_keys"].items():
                for target in devices:
                    key_id, key = self.one_time[(name, target)].popitem()
                    result.setdefault(name, {})[target] = {key_id: key}
            return httpx.Response(200, json={"one_time_keys": result, "failures": {}})
        if "/sendToDevice/" in path:
            assert "/m.room.encrypted/" in path
            for name, devices in body["messages"].items():
                for content in devices.values():
                    self.to_device[name].append(
                        {"sender": user, "type": "m.room.encrypted", "content": content}
                    )
            return httpx.Response(200, json={})
        if path.endswith("/joined_members"):
            return httpx.Response(
                200,
                json={
                    "joined": {
                        name: {"display_name": name, "avatar_url": None} for name in (ALICE, BOB)
                    }
                },
            )
        if path.endswith("/sync"):
            self.counter += 1
            state = [
                {
                    "type": "m.room.member",
                    "state_key": name,
                    "sender": name,
                    "event_id": "$member" + name,
                    "origin_server_ts": 1,
                    "content": {"membership": "join"},
                }
                for name in (ALICE, BOB)
            ]
            if self.encrypted:
                state.append(
                    {
                        "type": "m.room.encryption",
                        "state_key": "",
                        "sender": ALICE,
                        "event_id": "$encryption",
                        "origin_server_ts": 1,
                        "content": {"algorithm": "m.megolm.v1.aes-sha2"},
                    }
                )
            timeline, self.timeline[user] = self.timeline[user], []
            to_device, self.to_device[user] = self.to_device[user], []
            return httpx.Response(
                200,
                json={
                    "next_batch": str(self.counter),
                    "device_lists": {"changed": [ALICE, BOB], "left": []},
                    "device_one_time_keys_count": {
                        "signed_curve25519": len(self.one_time.get((user, device), {}))
                    },
                    "to_device": {"events": to_device},
                    "account_data": {
                        "events": [
                            {"type": "m.direct", "content": {ALICE if user == BOB else BOB: [ROOM]}}
                        ]
                    },
                    "rooms": {
                        "join": {
                            ROOM: {
                                "state": {"events": state},
                                "timeline": {
                                    "events": timeline,
                                    "limited": False,
                                    "prev_batch": "p",
                                },
                                "summary": {"m.joined_member_count": 2},
                                "ephemeral": {"events": []},
                                "account_data": {"events": []},
                            }
                        }
                    },
                },
            )
        if "/send/" in path:
            assert "/send/m.room.encrypted/" in path, "plaintext must never reach the network"
            self.counter += 1
            event = {
                "type": "m.room.encrypted",
                "sender": user,
                "content": body,
                "event_id": "$" + str(self.counter),
                "origin_server_ts": self.counter,
            }
            for name in (ALICE, BOB):
                self.timeline[name].append(event)
            return httpx.Response(200, json={"event_id": event["event_id"]})
        if path.endswith("/upload"):
            self.counter += 1
            key = str(self.counter)
            self.media[key] = request.content
            return httpx.Response(200, json={"content_uri": "mxc://matrix.invalid/" + key})
        if "/media/download/" in path:
            return httpx.Response(200, content=self.media[path.rsplit("/", 1)[-1]])
        raise AssertionError(f"Unexpected offline request {request.method} {path}")


async def pair(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MATRIX_PICKLE", "private-test-key-" + "k" * 40)
    server = Homeserver()
    clients = []
    stores = []
    for name, _user, peer in [("ALICE", ALICE, BOB), ("BOB", BOB, ALICE)]:
        config = ChannelConfig(
            homeserver="https://matrix.invalid",
            matrix_e2ee=True,
            matrix_device_id=name,
            matrix_store_path=str(tmp_path / name / "keys"),
            matrix_pickle_key_env="TEST_MATRIX_PICKLE",
            allowed_users=[peer],
            allowed_channels=[ROOM],
            profile=name,
        )
        client = MatrixTransport(
            config=config,
            token=name.lower() + "-token",
            client=httpx.AsyncClient(transport=httpx.MockTransport(server.request)),
        )
        await client.authenticate()
        clients.append(client)
        stores.append(ChannelStore(cwd=tmp_path / name, transport="matrix"))
    alice, bob = clients
    for client, peer in [(alice, bob), (bob, alice)]:
        assert (
            peer.crypto is not None
            and peer.crypto.client is not None
            and peer.crypto.client.olm is not None
        )
        client.config.matrix_trusted_devices[peer.bot_id] = {
            peer.config.matrix_device_id: peer.crypto.client.olm.account.identity_keys["ed25519"]
        }
    for client, store in zip(clients, stores, strict=True):
        await client.receive(store)
    return server, alice, bob, stores


async def cleanup(clients, stores):
    for client in clients:
        await client.close()
        await client.client.aclose()
    for store in stores:
        store.close()


async def test_real_crypto_encrypt_decrypt_restart_and_persistent_fingerprints(
    tmp_path, monkeypatch
):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        await alice.send(json.dumps([ROOM, ""]), "private synthetic message", "delivery-1")
        encrypted = [request for request in server.requests if "/send/" in request.url.path][-1]
        assert b"private synthetic message" not in encrypted.content
        await bob.receive(stores[1])
        received = stores[1].claim_message()
        assert received is not None and received.text == "private synthetic message"
        assert (
            bob.crypto is not None
            and bob.crypto.client is not None
            and bob.crypto.client.olm is not None
        )
        fingerprint = bob.crypto.client.olm.account.identity_keys.copy()
        await bob.close()
        await bob.authenticate()
        assert (
            bob.crypto is not None
            and bob.crypto.client is not None
            and bob.crypto.client.olm is not None
        )
        assert bob.crypto.client.olm.account.identity_keys == fingerprint
        # Alice reuses its Megolm session: Bob must restore the inbound session.
        await alice.send(json.dumps([ROOM, "$root"]), "after restart", "delivery-2")
        await bob.receive(stores[1])
        received = stores[1].claim_message()
        assert received is not None and received.text == "after restart"
        assert json.loads(received.thread_id) == [ROOM, "$root"]
        assert stores[1].get("matrix_encryption")["pending_events"] == 0
    finally:
        await cleanup([alice, bob], stores)


async def test_unknown_or_changed_devices_block_outbound_and_plaintext_is_ignored(
    tmp_path, monkeypatch
):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        alice.config.matrix_trusted_devices[BOB]["BOB"] = "wrong-fingerprint"
        with pytest.raises(ChannelError, match="unverified"):
            await alice.send(json.dumps([ROOM, ""]), "must not send", "untrusted")
        assert not any("/send/" in request.url.path for request in server.requests)
        server.timeline[BOB].append(
            {
                "type": "m.room.message",
                "sender": ALICE,
                "event_id": "$plain",
                "origin_server_ts": 99,
                "content": {"msgtype": "m.text", "body": "plaintext instruction"},
            }
        )
        await bob.receive(stores[1])
        assert stores[1].claim_message() is None
        assert alice.crypto is not None and alice.crypto.client is not None
        with pytest.raises(ChannelError, match="plaintext room send"):
            await alice.crypto.client._send(
                type, "PUT", "/_matrix/client/v3/rooms/room/send/m.room.message/no", "{}"
            )
        alice.crypto.client.rooms[ROOM].encrypted = False
        with pytest.raises(ChannelError, match="encrypted room"):
            await alice.send(json.dumps([ROOM, ""]), "must not downgrade", "downgrade")
    finally:
        await cleanup([alice, bob], stores)


async def test_encrypted_attachment_upload_download_and_integrity(tmp_path, monkeypatch):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        raw = b"private attachment bytes"
        attachment = MediaAttachment(
            kind="file",
            mime_type="text/plain",
            name="private-name.txt",
            data=base64.b64encode(raw).decode(),
        )
        await alice.send_media(json.dumps([ROOM, ""]), attachment, "media")
        upload = [request for request in server.requests if "/upload" in request.url.path][-1]
        assert upload.content != raw and "private-name.txt" not in str(upload.url)
        assert upload.headers["content-type"] == "application/octet-stream"
        await bob.receive(stores[1])
        received = stores[1].claim_message()
        assert received is not None and "encrypted_file" in received.attachments[0]
        prepared = await bob.prepare_media(received)
        assert base64.b64decode(prepared[0].data or "") == raw
        media_id = received.attachments[0]["url"].rsplit("/", 1)[-1]
        server.media[media_id] = b"tampered"
        with pytest.raises(ChannelError, match="integrity"):
            await bob.prepare_media(received)
    finally:
        await cleanup([alice, bob], stores)


async def test_late_key_ciphertext_survives_restart_without_plaintext_execution(
    tmp_path, monkeypatch
):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        await alice.send(json.dumps([ROOM, ""]), "late key", "late")
        keys, server.to_device[BOB] = server.to_device[BOB], []
        await bob.receive(stores[1])
        assert stores[1].claim_message() is None
        assert len(stores[1].get("matrix_pending_ciphertext")) == 1
        await bob.close()
        await bob.authenticate()
        server.to_device[BOB].extend(keys)
        await bob.receive(stores[1])
        received = stores[1].claim_message()
        assert received is not None and received.text == "late key"
        assert stores[1].get("matrix_pending_ciphertext") == []
    finally:
        await cleanup([alice, bob], stores)


async def test_store_lock_identity_and_missing_passphrase_fail_closed(tmp_path, monkeypatch):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        duplicate = MatrixTransport(config=bob.config, token=bob.token, client=bob.client)
        with pytest.raises(ChannelError, match="already in use"):
            await duplicate.authenticate()
        await duplicate.close()
        await bob.close()
        monkeypatch.delenv("TEST_MATRIX_PICKLE")
        with pytest.raises(ChannelError, match="pickle-key"):
            await bob.authenticate()
        monkeypatch.setenv("TEST_MATRIX_PICKLE", "wrong-secret-" + "x" * 40)
        with pytest.raises(ChannelError, match="original device store"):
            await bob.authenticate()
        monkeypatch.setenv("TEST_MATRIX_PICKLE", "private-test-key-" + "k" * 40)
        bob.config = replace(bob.config, matrix_store_path=str(tmp_path / "new-empty-store"))
        with pytest.raises(ChannelError, match="differ"):
            await bob.authenticate()
        assert not any("access_token" in request.url.params for request in server.requests)
    finally:
        await cleanup([alice, bob], stores)


def test_matrix_configuration_requires_explicit_encryption_and_trust_settings():
    with pytest.raises(ValueError, match="requires"):
        ChannelConfig.from_dict({"matrix_e2ee": True})
    with pytest.raises(ValueError, match="blanket trust"):
        ChannelConfig.from_dict({"matrix_unverified_policy": "trust_all"})
    with pytest.raises(ValueError, match="fingerprints"):
        ChannelConfig.from_dict({"matrix_trusted_devices": {"@peer:example": ["device"]}})


async def test_decrypted_pending_events_survive_failure_before_inbox_commit(tmp_path, monkeypatch):
    server, alice, bob, stores = await pair(tmp_path, monkeypatch)
    try:
        await alice.send(json.dumps([ROOM, ""]), "durable plaintext boundary", "durable")
        keys, server.to_device[BOB] = server.to_device[BOB], []
        await bob.receive(stores[1])
        server.to_device[BOB].extend(keys)
        original = bob._ingest_sync

        def fail(*args):
            raise RuntimeError("synthetic interruption before inbox persistence")

        monkeypatch.setattr(bob, "_ingest_sync", fail)
        with pytest.raises(RuntimeError, match="synthetic"):
            await bob.receive(stores[1])
        assert stores[1].get("matrix_pending_ciphertext")
        monkeypatch.setattr(bob, "_ingest_sync", original)
        await bob.close()
        await bob.authenticate()
        await bob.receive(stores[1])
        received = stores[1].claim_message()
        assert received is not None and received.text == "durable plaintext boundary"
    finally:
        await cleanup([alice, bob], stores)
