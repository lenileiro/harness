from __future__ import annotations

WHATSAPP_BRIDGE_PACKAGE_JSON = """{
  "name": "harness-whatsapp-bridge",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "dependencies": {
    "@hapi/boom": "^10.0.1",
    "@whiskeysockets/baileys": "^6.7.18",
    "express": "^4.21.2",
    "link-preview-js": "^3.1.0",
    "pino": "^9.9.0",
    "qrcode-terminal": "^0.12.0"
  }
}
"""

WHATSAPP_BRIDGE_JS = r"""#!/usr/bin/env node
import express from 'express';
import { spawn } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import path from 'path';
import qrcode from 'qrcode-terminal';
import pino from 'pino';
import { Boom } from '@hapi/boom';
import {
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeWASocket,
  useMultiFileAuthState,
} from '@whiskeysockets/baileys';

const args = process.argv.slice(2);

function getArg(name, fallback) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
}

const PORT = Number.parseInt(getArg('port', '8741'), 10);
const SESSION_DIR = getArg('session', path.join(process.env.HOME || '~', '.harness', 'gateway', 'whatsapp', 'session'));
const MODE = getArg('mode', process.env.HARNESS_WHATSAPP_MODE || 'self-chat');
const PAIR_ONLY = args.includes('--pair-only');
const REPLY_PREFIX = process.env.HARNESS_WHATSAPP_REPLY_PREFIX || '';
const WORKSPACE_CWD = process.env.HARNESS_WHATSAPP_WORKSPACE_CWD || process.cwd();
const UV_BIN = process.env.HARNESS_WHATSAPP_UV_BIN || 'uv';
const ENV_FILE = process.env.HARNESS_WHATSAPP_ENV_FILE || '';
const MAX_GATEWAY_CONCURRENCY = Math.max(1, Number.parseInt(process.env.HARNESS_WHATSAPP_MAX_CONCURRENCY || '1', 10) || 1);
const MAX_GATEWAY_QUEUE = Math.max(0, Number.parseInt(process.env.HARNESS_WHATSAPP_MAX_QUEUE || '3', 10) || 0);
const GATEWAY_CHILD_TIMEOUT_MS = Math.max(5000, Number.parseInt(process.env.HARNESS_WHATSAPP_CHILD_TIMEOUT_MS || '600000', 10) || 600000);
const GATEWAY_OUTPUT_LIMIT_BYTES = Math.max(65536, Number.parseInt(process.env.HARNESS_WHATSAPP_OUTPUT_LIMIT_BYTES || '262144', 10) || 262144);
const BRIDGE_STARTED_AT_MS = Date.now();
const STARTUP_REPLAY_GRACE_MS_RAW = Number.parseInt(process.env.HARNESS_WHATSAPP_STARTUP_REPLAY_GRACE_MS || '10000', 10);
const STARTUP_REPLAY_GRACE_MS = Number.isFinite(STARTUP_REPLAY_GRACE_MS_RAW) ? Math.max(0, STARTUP_REPLAY_GRACE_MS_RAW) : 10000;
const MAX_PROCESSED_MESSAGE_IDS = Math.max(50, Number.parseInt(process.env.HARNESS_WHATSAPP_PROCESSED_MESSAGE_CACHE || '500', 10) || 500);
const PROCESSED_MESSAGES_PATH = path.join(SESSION_DIR, 'processed-messages.json');
const ACTIVE_BRIDGE_PATH = path.join(SESSION_DIR, 'active-bridge.json');
const BRIDGE_INSTANCE_ID = `${process.pid}-${BRIDGE_STARTED_AT_MS}-${Math.random().toString(16).slice(2)}`;
const ALLOWED_USERS = (process.env.HARNESS_WHATSAPP_ALLOWED_USERS || '')
  .split(',')
  .map((value) => value.trim().replace(/[^\d*]/g, ''))
  .filter(Boolean);

mkdirSync(SESSION_DIR, { recursive: true });
markActiveBridgeInstance();

const logger = pino({ level: 'warn' });
const app = express();
app.use(express.json({ limit: '2mb' }));

let sock = null;
let connectionState = 'disconnected';
let connectionOpenedAtMs = 0;
const processedMessageIds = loadProcessedMessageIds();
const latestInboundTokenByChat = new Map();
let activeGatewayTasks = 0;
const pendingGatewayTasks = [];

function markActiveBridgeInstance() {
  try {
    mkdirSync(path.dirname(ACTIVE_BRIDGE_PATH), { recursive: true });
    writeFileSync(
      ACTIVE_BRIDGE_PATH,
      JSON.stringify({
        instance_id: BRIDGE_INSTANCE_ID,
        pid: process.pid,
        started_at_ms: BRIDGE_STARTED_AT_MS,
      }),
      'utf8',
    );
  } catch (error) {
    console.error('⚠️ Failed to mark active WhatsApp bridge instance:', error);
  }
}

function isActiveBridgeInstance() {
  try {
    if (!existsSync(ACTIVE_BRIDGE_PATH)) {
      return true;
    }
    const payload = JSON.parse(readFileSync(ACTIVE_BRIDGE_PATH, 'utf8'));
    return String(payload?.instance_id || '') === BRIDGE_INSTANCE_ID;
  } catch (error) {
    console.error('⚠️ Failed to read active WhatsApp bridge instance:', error);
    return true;
  }
}

function loadProcessedMessageIds() {
  try {
    if (!existsSync(PROCESSED_MESSAGES_PATH)) {
      return new Set();
    }
    const payload = JSON.parse(readFileSync(PROCESSED_MESSAGES_PATH, 'utf8'));
    const ids = Array.isArray(payload?.ids) ? payload.ids : Array.isArray(payload) ? payload : [];
    return new Set(
      ids
        .map((value) => String(value || '').trim())
        .filter(Boolean)
        .slice(-MAX_PROCESSED_MESSAGE_IDS),
    );
  } catch (error) {
    console.error('⚠️ Failed to load processed WhatsApp message cache:', error);
    return new Set();
  }
}

function persistProcessedMessageIds() {
  try {
    mkdirSync(path.dirname(PROCESSED_MESSAGES_PATH), { recursive: true });
    writeFileSync(
      PROCESSED_MESSAGES_PATH,
      JSON.stringify({ ids: Array.from(processedMessageIds).slice(-MAX_PROCESSED_MESSAGE_IDS) }),
      'utf8',
    );
  } catch (error) {
    console.error('⚠️ Failed to persist processed WhatsApp message cache:', error);
  }
}

function rememberProcessedMessageId(messageId) {
  const id = String(messageId || '').trim();
  if (!id) return;
  processedMessageIds.add(id);
  while (processedMessageIds.size > MAX_PROCESSED_MESSAGE_IDS) {
    const oldest = processedMessageIds.values().next().value;
    if (!oldest) break;
    processedMessageIds.delete(oldest);
  }
  persistProcessedMessageIds();
}

function rememberInboundNode(node) {
  rememberProcessedMessageId(node?.key?.id);
}

function normalizeChatId(value) {
  const raw = String(value || '').trim();
  if (!raw) return raw;
  if (raw.includes('@')) return raw;
  const digits = raw.replace(/[^\d]/g, '');
  return `${digits}@s.whatsapp.net`;
}

function formatMessage(message) {
  if (MODE !== 'self-chat') return String(message || '');
  if (!REPLY_PREFIX) return String(message || '');
  return `${REPLY_PREFIX}${String(message || '')}`;
}

function normalizedReplyText(value) {
  return String(value || '').normalize('NFKC').replace(/\r\n/g, '\n').trim();
}

function isHarnessReplyText(text) {
  const trimmed = normalizedReplyText(text);
  if (!trimmed) return false;
  const prefix = normalizedReplyText(REPLY_PREFIX);
  if (prefix && trimmed.startsWith(prefix)) {
    return true;
  }
  const prefixTitle = prefix ? prefix.split('\n', 1)[0].trim() : '';
  const firstLine = trimmed.split('\n', 1)[0].trim();
  return Boolean(prefixTitle && firstLine === prefixTitle);
}

function digitsOnly(value) {
  return String(value || '').replace(/[^\d]/g, '');
}

function identityDigits(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const beforeAt = raw.split('@', 1)[0] || '';
  const beforeDevice = beforeAt.split(':', 1)[0] || beforeAt;
  return beforeDevice.replace(/[^\d]/g, '');
}

function extractMessageText(node) {
  const msg = node?.message;
  if (!msg) return '';
  if (msg.deviceSentMessage?.message) {
    return extractMessageText({ message: msg.deviceSentMessage.message });
  }
  if (typeof msg.conversation === 'string' && msg.conversation) {
    return msg.conversation;
  }
  if (typeof msg.extendedTextMessage?.text === 'string' && msg.extendedTextMessage.text) {
    return msg.extendedTextMessage.text;
  }
  if (typeof msg.imageMessage?.caption === 'string' && msg.imageMessage.caption) {
    return msg.imageMessage.caption;
  }
  if (typeof msg.videoMessage?.caption === 'string' && msg.videoMessage.caption) {
    return msg.videoMessage.caption;
  }
  if (msg.ephemeralMessage?.message) {
    return extractMessageText({ message: msg.ephemeralMessage.message });
  }
  if (msg.viewOnceMessage?.message) {
    return extractMessageText({ message: msg.viewOnceMessage.message });
  }
  if (msg.viewOnceMessageV2?.message) {
    return extractMessageText({ message: msg.viewOnceMessageV2.message });
  }
  if (msg.editedMessage?.message) {
    return extractMessageText({ message: msg.editedMessage.message });
  }
  return '';
}

function ownIdentityCandidates() {
  return [
    identityDigits(sock?.user?.id),
    identityDigits(sock?.user?.lid),
  ].filter(Boolean);
}

function isAllowedInbound(chatId, key) {
  if (MODE === 'self-chat') {
    if (!chatId || chatId.endsWith('@g.us') || chatId === 'status@broadcast') {
      return false;
    }
    const candidates = [
      digitsOnly(chatId),
      digitsOnly(key?.participant),
      digitsOnly(key?.remoteJid),
    ].filter(Boolean);
    const ownIds = ownIdentityCandidates();
    if (candidates.some((value) => ownIds.includes(value))) {
      return true;
    }
    return candidates.some((value) => ALLOWED_USERS.includes(value));
  }
  if (ALLOWED_USERS.includes('*')) {
    return true;
  }
  const candidates = [
    digitsOnly(chatId),
    digitsOnly(key?.participant),
    digitsOnly(key?.remoteJid),
  ].filter(Boolean);
  return candidates.some((value) => ALLOWED_USERS.includes(value));
}

function inboundUserId(chatId, key) {
  return digitsOnly(key?.participant) || digitsOnly(chatId) || String(chatId || 'whatsapp-user');
}

function messageTimestampMs(node) {
  const raw = node?.messageTimestamp;
  if (raw == null) return 0;
  let numeric = Number(raw);
  if ((!Number.isFinite(numeric) || numeric <= 0) && typeof raw?.toNumber === 'function') {
    numeric = Number(raw.toNumber());
  }
  if ((!Number.isFinite(numeric) || numeric <= 0) && typeof raw?.toString === 'function') {
    const text = String(raw.toString()).trim();
    if (/^\d+$/.test(text)) {
      numeric = Number(text);
    }
  }
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  return numeric < 1000000000000 ? numeric * 1000 : numeric;
}

function startupReplayCutoffMs() {
  return Math.max(BRIDGE_STARTED_AT_MS, connectionOpenedAtMs || 0);
}

function isStartupReplayWindow() {
  return Date.now() - startupReplayCutoffMs() <= STARTUP_REPLAY_GRACE_MS;
}

function inboundMessageToken(node, text) {
  const messageId = String(node?.key?.id || '').trim();
  if (messageId) return messageId;
  return `${messageTimestampMs(node)}:${normalizedReplyText(text)}`;
}

function shouldIgnoreInbound(node, text) {
  const chatId = String(node?.key?.remoteJid || '');
  if (!chatId || chatId === 'status@broadcast') {
    return true;
  }
  const trimmed = String(text || '').trim();
  if (!trimmed) {
    rememberInboundNode(node);
    return true;
  }
  if (isHarnessReplyText(trimmed)) {
    rememberInboundNode(node);
    return true;
  }
  const messageId = String(node?.key?.id || '').trim();
  if (messageId && processedMessageIds.has(messageId)) {
    return true;
  }
  const timestamp = messageTimestampMs(node);
  if (node?.key?.fromMe && !timestamp) {
    rememberInboundNode(node);
    return true;
  }
  if (node?.key?.fromMe && isStartupReplayWindow()) {
    rememberInboundNode(node);
    return true;
  }
  if (timestamp && timestamp <= startupReplayCutoffMs()) {
    rememberInboundNode(node);
    return true;
  }
  if (!timestamp && isStartupReplayWindow()) {
    rememberInboundNode(node);
    return true;
  }
  return false;
}

function startTypingTicker(chatId) {
  let stopped = false;
  let timer = null;

  async function tick() {
    if (stopped || !sock || connectionState !== 'connected') {
      return;
    }
    try {
      await sock.sendPresenceUpdate('composing', chatId);
    } catch (error) {
      console.error('❌ Failed to send typing presence:', error);
    }
    if (!stopped) {
      timer = setTimeout(tick, 4000);
    }
  }

  void tick();

  return () => {
    stopped = true;
    if (timer) {
      clearTimeout(timer);
    }
    if (sock && connectionState === 'connected') {
      void sock.sendPresenceUpdate('paused', chatId).catch(() => {});
    }
  };
}

function parseDotenvValue(raw) {
  const value = String(raw || '').trim();
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function shouldImportDotenvKey(key) {
  return (
    key.startsWith('HARNESS_') ||
    key === 'OPENROUTER_API_KEY' ||
    key === 'OPENAI_API_KEY' ||
    key === 'ANTHROPIC_API_KEY' ||
    key === 'TAVILY_API_KEY'
  );
}

function dotenvValuesToPrefer() {
  if (!ENV_FILE || !existsSync(ENV_FILE)) {
    return {};
  }
  const text = readFileSync(ENV_FILE, 'utf8');
  const preferred = {};
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (!match) continue;
    const key = String(match[1] || '').trim();
    if (!key) continue;
    if (shouldImportDotenvKey(key)) {
      preferred[key] = parseDotenvValue(match[2] || '');
    }
  }
  return preferred;
}

function appendLimited(current, chunk) {
  const combined = current + chunk.toString();
  if (combined.length <= GATEWAY_OUTPUT_LIMIT_BYTES) {
    return combined;
  }
  return combined.slice(combined.length - GATEWAY_OUTPUT_LIMIT_BYTES);
}

function looksLikeGatewayPayload(value) {
  return Boolean(
    value &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      value.reply &&
      typeof value.reply === 'object' &&
      value.session &&
      typeof value.session === 'object',
  );
}

function extractBalancedJsonObject(source, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let idx = start; idx < source.length; idx += 1) {
    const ch = source[idx];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(start, idx + 1);
      }
    }
  }
  return '';
}

function lineObjectStarts(source) {
  const starts = [];
  let atLineStart = true;
  for (let idx = 0; idx < source.length; idx += 1) {
    if (atLineStart) {
      let start = idx;
      while (source[start] === ' ' || source[start] === '\t') {
        start += 1;
      }
      if (source[start] === '{') {
        starts.push(start);
      }
      atLineStart = false;
    }
    if (source[idx] === '\n' || source[idx] === '\r') {
      atLineStart = true;
    }
  }
  return starts;
}

function parseGatewayJsonOutput(stdout) {
  const raw = String(stdout || '').trim();
  if (!raw) {
    throw new Error('gateway command produced no stdout');
  }
  try {
    return JSON.parse(raw);
  } catch (firstError) {
    const starts = lineObjectStarts(raw);
    for (let idx = starts.length - 1; idx >= 0; idx -= 1) {
      const candidate = extractBalancedJsonObject(raw, starts[idx]);
      if (!candidate) continue;
      try {
        const payload = JSON.parse(candidate);
        if (looksLikeGatewayPayload(payload)) {
          return payload;
        }
      } catch (_) {
        // Keep looking for the final gateway payload after noisy log lines.
      }
    }
    throw firstError;
  }
}

function drainGatewayQueue() {
  while (activeGatewayTasks < MAX_GATEWAY_CONCURRENCY && pendingGatewayTasks.length > 0) {
    const item = pendingGatewayTasks.shift();
    activeGatewayTasks += 1;
    item
      .run()
      .then(item.resolve)
      .catch((error) => item.resolve({ ok: false, error: String(error) }))
      .finally(() => {
        activeGatewayTasks -= 1;
        drainGatewayQueue();
      });
  }
}

function enqueueGatewayTask(run) {
  if (activeGatewayTasks + pendingGatewayTasks.length >= MAX_GATEWAY_CONCURRENCY + MAX_GATEWAY_QUEUE) {
    return Promise.resolve({
      ok: false,
      error: 'gateway queue full',
    });
  }
  return new Promise((resolve) => {
    pendingGatewayTasks.push({ run, resolve });
    drainGatewayQueue();
  });
}

async function dispatchInboundCommand({ chatId, userId, text, messageId, messageToken }) {
  const receiveArgs = [
    'run',
    'harness',
    'gateway',
    'receive',
    '--cwd',
    WORKSPACE_CWD,
    '--message',
    text,
    '--transport',
    'whatsapp',
    '--user',
    userId,
    '--thread',
    chatId,
    '--json',
  ];
  async function runGateway(args) {
    console.log('🚀 gateway child', JSON.stringify({ cmd: UV_BIN, args }));
    const childEnv = { ...process.env };
    const dotenvValues = dotenvValuesToPrefer();
    for (const [key, value] of Object.entries(dotenvValues)) {
      childEnv[key] = value;
    }
    return await new Promise((resolve) => {
      const child = spawn(UV_BIN, args, {
        cwd: WORKSPACE_CWD,
        env: childEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      let settled = false;
      const timeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        console.error('❌ Gateway command timed out');
        child.kill('SIGTERM');
        setTimeout(() => child.kill('SIGKILL'), 5000).unref();
        resolve({ ok: false, error: `gateway command timed out after ${GATEWAY_CHILD_TIMEOUT_MS}ms` });
      }, GATEWAY_CHILD_TIMEOUT_MS);
      timeout.unref();
      child.stdout.on('data', (chunk) => {
        stdout = appendLimited(stdout, chunk);
      });
      child.stderr.on('data', (chunk) => {
        stderr = appendLimited(stderr, chunk);
      });
      child.on('error', (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        console.error('❌ Failed to run gateway command:', error);
        resolve({ ok: false, error: String(error) });
      });
      child.on('close', (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        console.log(
          '🧾 gateway child exit',
          JSON.stringify({
            code,
            stdoutBytes: stdout.length,
            stderr: stderr ? stderr.slice(-2000) : '',
          }),
        );
        if (code !== 0) {
          console.error('❌ Gateway command exited non-zero:', stderr || stdout);
          resolve({ ok: false, error: stderr || stdout || `exit ${code}` });
          return;
        }
        try {
          const payload = parseGatewayJsonOutput(stdout);
          resolve({ ok: true, payload });
        } catch (error) {
          console.error('❌ Failed to parse gateway command output:', stdout.slice(-4000));
          resolve({ ok: false, error: String(error) });
        }
      });
    });
  }
  const result = await runGateway(receiveArgs);
  console.log(
    '🗨️ gateway result',
    JSON.stringify({
      ok: Boolean(result?.ok),
      error: result?.error ? String(result.error).slice(0, 500) : '',
      command: result?.payload?.reply?.command || '',
      status: result?.payload?.reply?.status || '',
      textBytes: result?.payload?.reply?.text ? String(result.payload.reply.text).length : 0,
      sessionId: result?.payload?.session?.id || '',
    }),
  );
  if (!result?.ok) {
    return result;
  }
  return {
    ok: true,
    replyText: result.payload?.reply?.text
      ? String(result.payload.reply.text)
      : '',
    sessionId: result.payload?.session?.id || '',
    messageId,
    messageToken,
  };
}

async function startSocket() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    auth: state,
    logger,
    printQRInTerminal: false,
    browser: ['Harness', 'Chrome', '120.0'],
    syncFullHistory: false,
    fireInitQueries: false,
    markOnlineOnConnect: false,
    getMessage: async () => ({ conversation: '' }),
  });

  sock.ev.on('creds.update', saveCreds);
  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log('\n📱 Scan this QR with WhatsApp on your phone:\n');
      qrcode.generate(qr, { small: true });
      console.log('\nWaiting for scan...\n');
    }

    if (connection === 'open') {
      connectionState = 'connected';
      connectionOpenedAtMs = Date.now();
      console.log('✅ WhatsApp connected');
      if (PAIR_ONLY) {
        setTimeout(() => process.exit(0), 1500);
      }
      return;
    }

    if (connection === 'close') {
      connectionState = 'disconnected';
      const reason = new Boom(lastDisconnect?.error)?.output?.statusCode;
      if (reason === DisconnectReason.loggedOut) {
        console.error('❌ WhatsApp session logged out. Re-run pairing.');
        process.exit(1);
        return;
      }
      const delay = reason === 515 ? 1000 : 3000;
      setTimeout(startSocket, delay);
    }
  });
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    const upsertType = String(type || '').trim();
    if (upsertType && upsertType !== 'notify') {
      console.log('SKIP non-notify upsert', JSON.stringify({ type: upsertType, count: (messages || []).length }));
      return;
    }
    for (const node of messages || []) {
      const chatId = String(node?.key?.remoteJid || '');
      const text = extractMessageText(node);
      console.log(
        '📨 inbound',
        JSON.stringify({
          chatId,
          fromMe: Boolean(node?.key?.fromMe),
          participant: String(node?.key?.participant || ''),
          text,
        }),
      );
      if (shouldIgnoreInbound(node, text)) {
        console.log('SKIP ignore', JSON.stringify({ chatId, text }));
        continue;
      }
      const allowed = isAllowedInbound(chatId, node?.key);
      console.log(
        'ALLOW check',
        JSON.stringify({
          chatId,
          userId: inboundUserId(chatId, node?.key),
          allowed,
          ownIds: ownIdentityCandidates(),
          allowedUsers: ALLOWED_USERS,
        }),
      );
      if (!allowed) {
        console.log('SKIP disallowed', JSON.stringify({ chatId }));
        continue;
      }
      const messageId = String(node?.key?.id || '').trim();
      rememberProcessedMessageId(messageId);
      const userId = inboundUserId(chatId, node?.key);
      const messageToken = inboundMessageToken(node, text);
      latestInboundTokenByChat.set(chatId, messageToken);
      console.log('ENTER dispatch', JSON.stringify({ chatId, userId, text: String(text || '').trim() }));
      try {
        const result = await enqueueGatewayTask(async () => {
          const stopTyping = startTypingTicker(chatId);
          try {
            return await dispatchInboundCommand({
              chatId,
              userId,
              text: String(text || '').trim(),
              messageId: node?.key?.id || '',
              messageToken,
            });
          } finally {
            stopTyping();
          }
        });
        if (!result?.ok) {
          console.log('SKIP gateway task', JSON.stringify({ chatId, messageId, error: String(result?.error || '') }));
          continue;
        }
        if (!result.replyText) {
          continue;
        }
        if (result.messageToken && latestInboundTokenByChat.get(chatId) !== result.messageToken) {
          console.log('SKIP stale reply', JSON.stringify({ chatId, messageId: result.messageId }));
          continue;
        }
        if (!isActiveBridgeInstance()) {
          console.log('SKIP inactive bridge reply', JSON.stringify({ chatId, messageId: result.messageId }));
          continue;
        }
        await sock.sendMessage(chatId, { text: formatMessage(result.replyText) });
      } catch (error) {
        console.error('❌ Failed to send gateway reply:', error);
      }
    }
  });
}

app.get('/health', (_req, res) => {
  res.json({
    status: connectionState,
    mode: MODE,
    paired: connectionState === 'connected',
    user: sock?.user || null,
    workspace_cwd: WORKSPACE_CWD,
    session_dir: SESSION_DIR,
    active_gateway_tasks: activeGatewayTasks,
    pending_gateway_tasks: pendingGatewayTasks.length,
    max_gateway_concurrency: MAX_GATEWAY_CONCURRENCY,
    max_gateway_queue: MAX_GATEWAY_QUEUE,
    gateway_child_timeout_ms: GATEWAY_CHILD_TIMEOUT_MS,
    startup_replay_grace_ms: STARTUP_REPLAY_GRACE_MS,
  });
});

app.post('/send', async (req, res) => {
  if (!sock || connectionState !== 'connected') {
    res.status(503).json({ error: 'whatsapp bridge is not connected' });
    return;
  }

  const chatId = normalizeChatId(req.body?.chatId);
  const message = String(req.body?.message || '').trim();
  if (!chatId || !message) {
    res.status(400).json({ error: 'chatId and message are required' });
    return;
  }
  if (!isActiveBridgeInstance()) {
    res.status(409).json({ error: 'whatsapp bridge instance is no longer active' });
    return;
  }

  try {
    const payload = { text: formatMessage(message) };
    const sent = await sock.sendMessage(chatId, payload);
    res.json({
      ok: true,
      chatId,
      messageId: sent?.key?.id || null,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post('/typing', async (req, res) => {
  if (!sock || connectionState !== 'connected') {
    res.status(503).json({ error: 'whatsapp bridge is not connected' });
    return;
  }

  const chatId = normalizeChatId(req.body?.chatId);
  if (!chatId) {
    res.status(400).json({ error: 'chatId is required' });
    return;
  }

  try {
    await sock.sendPresenceUpdate('composing', chatId);
    res.json({ ok: true, chatId });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.listen(PORT, async () => {
  console.log(`Harness WhatsApp bridge listening on http://127.0.0.1:${PORT}`);
  await startSocket();
});
"""


__all__ = ["WHATSAPP_BRIDGE_JS", "WHATSAPP_BRIDGE_PACKAGE_JSON"]
