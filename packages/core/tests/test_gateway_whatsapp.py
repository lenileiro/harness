from __future__ import annotations

import json
import shutil
import signal
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.core.gateway_whatsapp import (
    DEFAULT_GATEWAY_CHILD_TIMEOUT_SECONDS,
    WhatsAppBridgeConfig,
    build_whatsapp_bridge_env,
    ensure_whatsapp_bridge_project,
    load_whatsapp_bridge_config,
    read_whatsapp_bridge_status,
    save_whatsapp_bridge_config,
    send_whatsapp_text_message,
    start_whatsapp_bridge,
    stop_stale_whatsapp_bridge,
)


def test_whatsapp_bridge_config_roundtrip(tmp_path: Path) -> None:
    config = WhatsAppBridgeConfig(
        enabled=True,
        provider="ollama",
        model="gemma4:latest",
        mode="self-chat",
        allowed_users=["15551234567"],
        bridge_port=9901,
        reply_prefix="Harness\n",
    )
    save_whatsapp_bridge_config(tmp_path, config)

    loaded = load_whatsapp_bridge_config(tmp_path)
    assert loaded == config


def test_ensure_whatsapp_bridge_project_writes_assets(tmp_path: Path) -> None:
    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    assert (project_dir / "package.json").exists()
    assert (project_dir / "bridge.js").exists()
    package_json = (project_dir / "package.json").read_text(encoding="utf-8")
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    assert "link-preview-js" in package_json
    assert "messages.upsert" in bridge_js
    assert "async ({ messages, type })" in bridge_js
    assert "'receive'" in bridge_js
    assert "'dispatch'" not in bridge_js
    assert "'converse'" not in bridge_js
    assert "ownIdentityCandidates" in bridge_js
    assert "chatId.endsWith('@g.us')" in bridge_js
    assert "BRIDGE_STARTED_AT_MS" in bridge_js
    assert "connectionOpenedAtMs = Date.now()" in bridge_js
    assert "startupReplayCutoffMs" in bridge_js
    assert "timestamp <= startupReplayCutoffMs()" in bridge_js
    assert "timestamp <= BRIDGE_STARTED_AT_MS" not in bridge_js
    assert "timestamp < BRIDGE_STARTED_AT_MS - 5000" not in bridge_js
    assert "processedMessageIds" in bridge_js
    assert "PROCESSED_MESSAGES_PATH" in bridge_js
    assert "loadProcessedMessageIds()" in bridge_js
    assert "writeFileSync" in bridge_js
    assert "fireInitQueries: false" in bridge_js
    assert "latestInboundTokenByChat" in bridge_js
    assert "inboundMessageToken" in bridge_js
    assert "isHarnessReplyText" in bridge_js
    assert "MAX_GATEWAY_CONCURRENCY" in bridge_js
    assert "pendingGatewayTasks" in bridge_js
    assert "enqueueGatewayTask" in bridge_js
    assert "ACTIVE_BRIDGE_PATH" in bridge_js
    assert "BRIDGE_INSTANCE_ID" in bridge_js
    assert "isActiveBridgeInstance" in bridge_js
    assert "SKIP inactive bridge reply" in bridge_js
    legacy_restart_text = "Harness is still working through " + "earlier WhatsApp messages"
    assert legacy_restart_text not in bridge_js
    assert "SKIP gateway task" in bridge_js
    assert "SKIP non-notify upsert" in bridge_js
    assert "SKIP stale reply" in bridge_js
    assert "GATEWAY_CHILD_TIMEOUT_MS" in bridge_js
    assert "600000" in bridge_js
    assert "GATEWAY_OUTPUT_LIMIT_BYTES" in bridge_js
    assert "STARTUP_REPLAY_GRACE_MS" in bridge_js
    assert "dotenvValuesToPrefer" in bridge_js
    assert "childEnv[key] = value" in bridge_js
    assert "stdoutBytes" in bridge_js
    assert "parseGatewayJsonOutput(stdout)" in bridge_js
    assert "looksLikeGatewayPayload" in bridge_js
    assert "extractBalancedJsonObject" in bridge_js
    assert "gateway result" in bridge_js
    assert "falling back to converse" not in bridge_js
    assert "workspace_cwd" in bridge_js
    assert "active_gateway_tasks" in bridge_js


def test_whatsapp_bridge_parses_gateway_json_after_log_noise(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function looksLikeGatewayPayload")
    end = bridge_js.index("function drainGatewayQueue")
    helper_block = bridge_js[start:end]

    payload = {
        "reply": {"command": "chat", "status": "ok", "text": "hello"},
        "session": {"id": "gw-test"},
    }
    noisy_stdout = (
        "2026-05-31T11:29:54.668891Z [warning  ] agent.step.error error='boom'\n"
        '{"level":"warn","msg":"not the gateway payload"}\n' + json.dumps(payload, indent=2) + "\n"
    )
    script = (
        helper_block
        + "\n"
        + f"const parsed = parseGatewayJsonOutput({json.dumps(noisy_stdout)});\n"
        + "if (parsed.reply.text !== 'hello' || parsed.session.id !== 'gw-test') {\n"
        + "  console.error(JSON.stringify(parsed));\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_parses_gateway_json_after_openrouter_error_noise(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function looksLikeGatewayPayload")
    end = bridge_js.index("function drainGatewayQueue")
    helper_block = bridge_js[start:end]

    payload = {
        "reply": {
            "session_id": "gw-whatsapp-thread",
            "command": "chat",
            "status": "ok",
            "text": "Workflow failed before a verified final report was produced.",
            "data": {"workflow_status": "failed"},
        },
        "session": {"id": "gw-whatsapp-thread"},
    }
    noisy_stdout = (
        "2026-05-31T11:29:54.668891Z [warning  ] agent.step.error "
        'error=\'OpenRouter HTTP 400. Body: {"error":{"message":"Provider '
        'returned error","code":400,"metadata":{"raw":"{\\"error\\":'
        '{\\"message\\":\\"System message must be at the beginning.\\"}}"}}}\'\n'
        "2026-05-31T11:29:54.668999Z [error    ] agent.run.failed "
        'error=\'OpenRouter HTTP 400. Body: {"error":{"message":"Provider returned error"}}\'\n'
        + json.dumps(payload, indent=2)
        + "\n"
    )
    script = (
        helper_block
        + "\n"
        + f"const parsed = parseGatewayJsonOutput({json.dumps(noisy_stdout)});\n"
        + "if (parsed.reply.command !== 'chat' || parsed.session.id !== 'gw-whatsapp-thread') {\n"
        + "  console.error(JSON.stringify(parsed));\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_ignores_replayed_messages_from_before_restart(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n------------\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 0;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const stale = { key: { remoteJid: '15551234567@s.whatsapp.net', id: 'old' }, messageTimestamp: 199 };\n"
        + "const fresh = { key: { remoteJid: '15551234567@s.whatsapp.net', id: 'new' }, messageTimestamp: 201 };\n"
        + "if (!shouldIgnoreInbound(stale, 'hello')) {\n"
        + "  console.error('stale restart message was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "if (shouldIgnoreInbound(fresh, 'hello')) {\n"
        + "  console.error('fresh post-start message was ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_ignores_replayed_messages_with_protobuf_timestamp(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n------------\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 0;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const stale = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'old' },\n"
        + "  messageTimestamp: { toNumber: () => 199 },\n"
        + "};\n"
        + "const fresh = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'new' },\n"
        + "  messageTimestamp: { toString: () => '201' },\n"
        + "};\n"
        + "if (!shouldIgnoreInbound(stale, 'hello')) {\n"
        + "  console.error('stale object-timestamp restart message was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "if (shouldIgnoreInbound(fresh, 'hello')) {\n"
        + "  console.error('fresh object-timestamp message was ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_ignores_timestampless_messages_during_startup_grace(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n------------\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 0;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const missingTimestamp = { key: { remoteJid: '15551234567@s.whatsapp.net', id: 'missing' } };\n"
        + "const freshMissingTimestamp = { key: { remoteJid: '15551234567@s.whatsapp.net', id: 'fresh-missing' } };\n"
        + "const fromMeMissingTimestamp = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'from-me-missing', fromMe: true },\n"
        + "};\n"
        + "Date.now = () => 205000;\n"
        + "if (!shouldIgnoreInbound(missingTimestamp, 'hello')) {\n"
        + "  console.error('timestampless startup replay was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "Date.now = () => 211000;\n"
        + "if (shouldIgnoreInbound(freshMissingTimestamp, 'hello')) {\n"
        + "  console.error('timestampless post-grace message was ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "if (!shouldIgnoreInbound(fromMeMissingTimestamp, 'hello')) {\n"
        + "  console.error('timestampless fromMe replay was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_persists_processed_message_ids_across_restart(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function loadProcessedMessageIds")
    end = bridge_js.index("function normalizeChatId")
    helper_block = bridge_js[start:end]
    cache_path = tmp_path / "session" / "processed-messages.json"
    script = (
        "import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';\n"
        "import path from 'path';\n"
        f"const PROCESSED_MESSAGES_PATH = {json.dumps(str(cache_path))};\n"
        "const MAX_PROCESSED_MESSAGE_IDS = 3;\n"
        "let processedMessageIds = loadProcessedMessageIds();\n"
        + helper_block
        + "\n"
        + "rememberProcessedMessageId('first');\n"
        + "rememberProcessedMessageId('second');\n"
        + "processedMessageIds = loadProcessedMessageIds();\n"
        + "if (!processedMessageIds.has('first') || !processedMessageIds.has('second')) {\n"
        + "  console.error('processed message cache did not survive reload');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "rememberProcessedMessageId('third');\n"
        + "rememberProcessedMessageId('fourth');\n"
        + "processedMessageIds = loadProcessedMessageIds();\n"
        + "if (processedMessageIds.has('first') || processedMessageIds.size !== 3) {\n"
        + "  console.error('processed message cache did not trim oldest ids');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_ignores_from_me_messages_during_startup_grace(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n------------\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 0;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const fromMeDuringStartup = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'skewed', fromMe: true },\n"
        + "  messageTimestamp: 201,\n"
        + "};\n"
        + "const fromMeAfterStartup = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'fresh', fromMe: true },\n"
        + "  messageTimestamp: 211,\n"
        + "};\n"
        + "Date.now = () => 205000;\n"
        + "if (!shouldIgnoreInbound(fromMeDuringStartup, 'hello')) {\n"
        + "  console.error('fromMe startup replay was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "Date.now = () => 211000;\n"
        + "if (shouldIgnoreInbound(fromMeAfterStartup, 'hello')) {\n"
        + "  console.error('fresh post-grace fromMe message was ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_replay_grace_starts_when_connection_opens(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n------------\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 260000;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const replayAfterSlowConnect = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'replay', fromMe: true },\n"
        + "  messageTimestamp: 261,\n"
        + "};\n"
        + "const freshAfterConnectionGrace = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'fresh', fromMe: true },\n"
        + "  messageTimestamp: 271,\n"
        + "};\n"
        + "Date.now = () => 265000;\n"
        + "if (!shouldIgnoreInbound(replayAfterSlowConnect, 'old message')) {\n"
        + "  console.error('connection-open replay was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "Date.now = () => 271000;\n"
        + "if (shouldIgnoreInbound(freshAfterConnectionGrace, 'new message')) {\n"
        + "  console.error('post-connection-grace message was ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_queue_overflow_is_silent(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function drainGatewayQueue")
    end = bridge_js.index("async function dispatchInboundCommand")
    helper_block = bridge_js[start:end]
    script = (
        "const MAX_GATEWAY_CONCURRENCY = 1;\n"
        "const MAX_GATEWAY_QUEUE = 0;\n"
        "let activeGatewayTasks = 1;\n"
        "const pendingGatewayTasks = [];\n"
        + helper_block
        + "\n"
        + "const result = await enqueueGatewayTask(async () => ({ ok: true, replyText: 'late' }));\n"
        + "if (result.ok || result.error !== 'gateway queue full' || Object.prototype.hasOwnProperty.call(result, 'replyText')) {\n"
        + "  console.error(JSON.stringify(result));\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_ignores_replayed_harness_reply_text(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function normalizedReplyText")
    end = bridge_js.index("function startTypingTicker")
    helper_block = bridge_js[start:end]
    script = (
        "const REPLY_PREFIX = 'Harness Agent\\n────────────\\n';\n"
        "const BRIDGE_STARTED_AT_MS = 200000;\n"
        "let connectionOpenedAtMs = 200000;\n"
        "const STARTUP_REPLAY_GRACE_MS = 10000;\n"
        "const processedMessageIds = new Set();\n"
        "function rememberProcessedMessageId(messageId) { if (messageId) processedMessageIds.add(String(messageId)); }\n"
        "function rememberInboundNode(node) { rememberProcessedMessageId(node?.key?.id); }\n"
        + helper_block
        + "\n"
        + "const replayedHarnessReply = {\n"
        + "  key: { remoteJid: '15551234567@s.whatsapp.net', id: 'harness-replay', fromMe: true },\n"
        + "  messageTimestamp: 201,\n"
        + "};\n"
        + "Date.now = () => 211000;\n"
        + "if (!shouldIgnoreInbound(replayedHarnessReply, 'Harness Agent\\n────────────\\nPreviously generated reply')) {\n"
        + "  console.error('Harness-owned replayed reply was not ignored');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_whatsapp_bridge_stale_instance_becomes_non_sendable(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function markActiveBridgeInstance")
    end = bridge_js.index("function loadProcessedMessageIds")
    helper_block = bridge_js[start:end]
    active_bridge_path = tmp_path / "session" / "active-bridge.json"
    script = (
        "import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';\n"
        "import path from 'path';\n"
        f"const ACTIVE_BRIDGE_PATH = {json.dumps(str(active_bridge_path))};\n"
        "const BRIDGE_STARTED_AT_MS = 123;\n"
        "const BRIDGE_INSTANCE_ID = 'first-bridge';\n"
        + helper_block
        + "\n"
        + "markActiveBridgeInstance();\n"
        + "if (!isActiveBridgeInstance()) {\n"
        + "  console.error('fresh bridge was not active');\n"
        + "  process.exit(1);\n"
        + "}\n"
        + "writeFileSync(\n"
        + "  ACTIVE_BRIDGE_PATH,\n"
        + "  JSON.stringify({ instance_id: 'replacement-bridge', pid: 999 }),\n"
        + "  'utf8',\n"
        + ");\n"
        + "if (isActiveBridgeInstance()) {\n"
        + "  console.error('stale bridge remained active');\n"
        + "  process.exit(1);\n"
        + "}\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_start_whatsapp_bridge_stops_previous_bridge_process(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(tmp_path, WhatsAppBridgeConfig(enabled=True))
    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    active_bridge_path = (
        tmp_path / ".harness" / "gateway" / "whatsapp" / "session" / "active-bridge.json"
    )
    active_bridge_path.parent.mkdir(parents=True, exist_ok=True)
    active_bridge_path.write_text(json.dumps({"pid": 43210}), encoding="utf-8")
    started: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["ps", "-p", "43210"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=f"node {project_dir / 'bridge.js'} --port 8741\n",
                stderr="",
            )
        if args[:2] == ["ps", "-axo"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        started.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        patch("harness.core.gateway_whatsapp.subprocess.run", side_effect=fake_run),
        patch("harness.core.gateway_whatsapp._process_exists", return_value=False),
        patch("harness.core.gateway_whatsapp.os.kill") as kill,
    ):
        start_whatsapp_bridge(tmp_path, node_bin="node")

    kill.assert_called_once_with(43210, signal.SIGTERM)
    assert any(command[:2] == ["node", str(project_dir / "bridge.js")] for command in started)


def test_start_whatsapp_bridge_stops_unmarked_previous_bridge_process(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(tmp_path, WhatsAppBridgeConfig(enabled=True))
    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    started: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["ps", "-axo"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=(
                    "101 /usr/bin/python unrelated.py\n"
                    f"43211 node {project_dir / 'bridge.js'} --port 8741\n"
                    "43212 node /tmp/other-harness/bridge.js --port 8742\n"
                ),
                stderr="",
            )
        started.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        patch("harness.core.gateway_whatsapp.subprocess.run", side_effect=fake_run),
        patch("harness.core.gateway_whatsapp._process_exists", return_value=False),
        patch("harness.core.gateway_whatsapp.os.kill") as kill,
    ):
        start_whatsapp_bridge(tmp_path, node_bin="node")

    kill.assert_called_once_with(43211, signal.SIGTERM)
    assert any(command[:2] == ["node", str(project_dir / "bridge.js")] for command in started)


def test_stop_stale_whatsapp_bridge_ignores_unrelated_process(tmp_path: Path) -> None:
    active_bridge_path = (
        tmp_path / ".harness" / "gateway" / "whatsapp" / "session" / "active-bridge.json"
    )
    active_bridge_path.parent.mkdir(parents=True, exist_ok=True)
    active_bridge_path.write_text(json.dumps({"pid": 43210}), encoding="utf-8")

    with (
        patch("harness.core.gateway_whatsapp._process_command", return_value="python unrelated.py"),
        patch("harness.core.gateway_whatsapp._discover_whatsapp_bridge_pids", return_value=[]),
        patch("harness.core.gateway_whatsapp.os.kill") as kill,
    ):
        stopped = stop_stale_whatsapp_bridge(tmp_path)

    assert stopped is False
    kill.assert_not_called()


def test_build_whatsapp_bridge_env_includes_workspace_and_uv(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=["15551234567"],
            bridge_port=8741,
            max_gateway_concurrency=1,
            max_gateway_queue=2,
            gateway_child_timeout_seconds=90,
        ),
    )
    env = build_whatsapp_bridge_env(tmp_path)
    assert env["HARNESS_WHATSAPP_MODE"] == "self-chat"
    assert env["HARNESS_WHATSAPP_ALLOWED_USERS"] == "15551234567"
    assert env["HARNESS_WHATSAPP_WORKSPACE_CWD"] == str(tmp_path.resolve())
    assert env["HARNESS_WHATSAPP_UV_BIN"]
    assert env["HARNESS_WHATSAPP_ENV_FILE"] == ""
    assert env["HARNESS_WHATSAPP_MAX_CONCURRENCY"] == "1"
    assert env["HARNESS_WHATSAPP_MAX_QUEUE"] == "2"
    assert env["HARNESS_WHATSAPP_CHILD_TIMEOUT_MS"] == "90000"
    assert env["HARNESS_SHELL_DEFAULT_TIMEOUT"] == "30"


def test_whatsapp_bridge_default_timeout_allows_long_workflows(tmp_path: Path) -> None:
    config = WhatsAppBridgeConfig()
    assert config.gateway_child_timeout_seconds == DEFAULT_GATEWAY_CHILD_TIMEOUT_SECONDS
    save_whatsapp_bridge_config(tmp_path, config)

    env = build_whatsapp_bridge_env(tmp_path)

    assert env["HARNESS_WHATSAPP_CHILD_TIMEOUT_MS"] == "600000"


def test_build_whatsapp_bridge_env_exposes_dotenv_path_when_present(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
            bridge_port=8741,
        ),
    )
    env = build_whatsapp_bridge_env(tmp_path)
    assert env["HARNESS_WHATSAPP_ENV_FILE"] == str((tmp_path / ".env").resolve())


def test_whatsapp_bridge_imports_harness_dotenv_settings_for_child_process(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=test-key",
                "HARNESS_OPENROUTER_MODEL_FALLBACKS=openai/gpt-4.1-mini,google/gemini-2.5-flash-lite",
                "PATH=/should/not/import",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    start = bridge_js.index("function parseDotenvValue")
    end = bridge_js.index("function appendLimited")
    helper_block = bridge_js[start:end]
    script = (
        "import { existsSync, readFileSync } from 'fs';\n"
        + f"const ENV_FILE = {json.dumps(str(dotenv_path))};\n"
        + helper_block
        + "\n"
        + "const values = dotenvValuesToPrefer();\n"
        + "if (values.OPENROUTER_API_KEY !== 'test-key') process.exit(1);\n"
        + "if (values.HARNESS_OPENROUTER_MODEL_FALLBACKS !== 'openai/gpt-4.1-mini,google/gemini-2.5-flash-lite') process.exit(2);\n"
        + "if (Object.prototype.hasOwnProperty.call(values, 'PATH')) process.exit(3);\n"
    )

    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_read_whatsapp_bridge_status_reports_defaults(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=False,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=19841,
        ),
    )
    status = read_whatsapp_bridge_status(tmp_path)
    assert status.config.mode == "self-chat"
    assert status.paired is False
    assert status.bridge_running is False
    assert status.bridge_connected is False


def test_probe_whatsapp_bridge_rejects_other_workspace_on_same_port(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=9913,
        ),
    )

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "status": "connected",
                    "workspace_cwd": str((tmp_path / "other").resolve()),
                }
            ).encode("utf-8")

    with patch("harness.core.gateway_whatsapp.request.urlopen", lambda req, timeout=0: _Response()):
        status = read_whatsapp_bridge_status(tmp_path)

    assert status.bridge_running is False
    assert status.bridge_connected is False


def test_send_whatsapp_text_message_posts_to_local_bridge(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=9912,
        ),
    )
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            if captured.get("health_checked"):
                return json.dumps({"ok": True, "messageId": "wamid.local"}).encode("utf-8")
            captured["health_checked"] = True
            return json.dumps(
                {
                    "status": "connected",
                    "workspace_cwd": str(tmp_path.resolve()),
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        if req.data is not None:
            captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    with patch("harness.core.gateway_whatsapp.request.urlopen", _fake_urlopen):
        payload = send_whatsapp_text_message(cwd=tmp_path, to="15551234567", text="hello")

    assert payload["ok"] is True
    assert captured["url"] == "http://127.0.0.1:9912/send"
    assert captured["body"] == {
        "chatId": "15551234567",
        "message": "hello",
    }
