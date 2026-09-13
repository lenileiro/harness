from __future__ import annotations

import shutil
import subprocess

import pytest

from harness.core.gateway_whatsapp_assets import WHATSAPP_BRIDGE_JS


def test_whatsapp_bridge_downloads_and_sends_bounded_media_with_real_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    start = WHATSAPP_BRIDGE_JS.index("function mediaNode(")
    end = WHATSAPP_BRIDGE_JS.index("function ownIdentityCandidates(")
    script = (
        """
import path from 'node:path';
import assert from 'node:assert/strict';
const sent = [];
const sock = { updateMediaMessage: () => {}, sendMessage: async (chatId,payload) => { sent.push({chatId,payload}); } };
const pino = () => ({});
const formatMessage = (text) => text;
const downloadMediaMessage = async () => (async function* () { yield Buffer.from('image'); })();
"""
        + WHATSAPP_BRIDGE_JS[start:end]
        + """
const media = await inboundMedia({message:{imageMessage:{mimetype:'image/png',fileLength:5,fileName:'image.png'}}});
assert.equal(media[0].data, Buffer.from('image').toString('base64'));
assert.equal(media[0].kind, 'image');
assert.equal(mediaNode({message:{viewOnceMessage:{message:{imageMessage:{}}}}}),null);
await sendGatewayReply('original-thread', '🦜'.repeat(2500), media);
assert.equal(sent.length, 3);
assert.equal(sent[0].payload.text + sent[1].payload.text, '🦜'.repeat(2500));
assert.equal(sent[0].payload.text.length, 4000);
assert.equal(sent[2].chatId, 'original-thread');
assert.equal(sent[2].payload.image.toString(), 'image');
assert.throws(() => mediaSendPayload({kind:'image',mime_type:'image/png',url:'file:///private'}));
await assert.rejects(inboundMedia({message:{imageMessage:{fileLength:30*1024*1024}}}));
"""
    )
    result = subprocess.run(
        [node, "--input-type=module"], input=script, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_whatsapp_bridge_media_uses_stdin_without_process_argument_payload(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    target = tmp_path / "bridge.mjs"
    target.write_text(WHATSAPP_BRIDGE_JS, encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(target)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "child.stdin.end(JSON.stringify(attachments))" in WHATSAPP_BRIDGE_JS
    assert "'--attachments-stdin'" in WHATSAPP_BRIDGE_JS
