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
import { mkdirSync } from 'fs';
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
const BRIDGE_STARTED_AT_MS = Date.now();
const ALLOWED_USERS = (process.env.HARNESS_WHATSAPP_ALLOWED_USERS || '')
  .split(',')
  .map((value) => value.trim().replace(/[^\d*]/g, ''))
  .filter(Boolean);

mkdirSync(SESSION_DIR, { recursive: true });

const logger = pino({ level: 'warn' });
const app = express();
app.use(express.json({ limit: '2mb' }));

let sock = null;
let connectionState = 'disconnected';
const processedMessageIds = new Set();

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
  const numeric = Number(raw);
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  return numeric < 1000000000000 ? numeric * 1000 : numeric;
}

function shouldIgnoreInbound(node, text) {
  const chatId = String(node?.key?.remoteJid || '');
  if (!chatId || chatId === 'status@broadcast') {
    return true;
  }
  const messageId = String(node?.key?.id || '').trim();
  if (messageId && processedMessageIds.has(messageId)) {
    return true;
  }
  const timestamp = messageTimestampMs(node);
  if (timestamp && timestamp < BRIDGE_STARTED_AT_MS - 5000) {
    return true;
  }
  const trimmed = String(text || '').trim();
  if (!trimmed) {
    return true;
  }
  if (REPLY_PREFIX && trimmed.startsWith(REPLY_PREFIX.trim())) {
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

async function dispatchInboundCommand({ chatId, userId, text, messageId }) {
  const dispatchArgs = [
    'run',
    'harness',
    'gateway',
    'dispatch',
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
  const converseArgs = [
    'run',
    'harness',
    'gateway',
    'converse',
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
    return await new Promise((resolve) => {
      const child = spawn(UV_BIN, args, {
        cwd: WORKSPACE_CWD,
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on('data', (chunk) => {
        stderr += chunk.toString();
      });
      child.on('error', (error) => {
        console.error('❌ Failed to run gateway command:', error);
        resolve({ ok: false, error: String(error) });
      });
      child.on('close', (code) => {
        console.log(
          '🧾 gateway child exit',
          JSON.stringify({
            code,
            stdout,
            stderr,
          }),
        );
        if (code !== 0) {
          console.error('❌ Gateway command exited non-zero:', stderr || stdout);
          resolve({ ok: false, error: stderr || stdout || `exit ${code}` });
          return;
        }
        try {
          const payload = JSON.parse(stdout);
          resolve({ ok: true, payload });
        } catch (error) {
          console.error('❌ Failed to parse gateway command output:', stdout);
          resolve({ ok: false, error: String(error) });
        }
      });
    });
  }
  const dispatched = await runGateway(dispatchArgs);
  console.log('📬 dispatch result', JSON.stringify(dispatched));
  if (dispatched?.ok && dispatched?.payload?.reply?.command !== 'unknown') {
    return {
      ok: true,
      replyText: dispatched.payload?.reply?.text ? String(dispatched.payload.reply.text) : '',
      sessionId: dispatched.payload?.session?.id || '',
      messageId,
    };
  }
  console.log('💬 falling back to converse');
  const conversational = await runGateway(converseArgs);
  console.log('🗨️ converse result', JSON.stringify(conversational));
  if (!conversational?.ok) {
    return conversational;
  }
  return {
    ok: true,
    replyText: conversational.payload?.reply?.text
      ? String(conversational.payload.reply.text)
      : '',
    sessionId: conversational.payload?.session?.id || '',
    messageId,
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
  sock.ev.on('messages.upsert', async ({ messages }) => {
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
      if (messageId) {
        processedMessageIds.add(messageId);
        if (processedMessageIds.size > 200) {
          const oldest = processedMessageIds.values().next().value;
          if (oldest) {
            processedMessageIds.delete(oldest);
          }
        }
      }
      const userId = inboundUserId(chatId, node?.key);
      console.log('ENTER dispatch', JSON.stringify({ chatId, userId, text: String(text || '').trim() }));
      const stopTyping = startTypingTicker(chatId);
      try {
        const result = await dispatchInboundCommand({
          chatId,
          userId,
          text: String(text || '').trim(),
          messageId: node?.key?.id || '',
        });
        if (!result?.ok || !result.replyText) {
          continue;
        }
        await sock.sendMessage(chatId, { text: formatMessage(result.replyText) });
      } catch (error) {
        console.error('❌ Failed to send gateway reply:', error);
      } finally {
        stopTyping();
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
