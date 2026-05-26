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
    "pino": "^9.9.0",
    "qrcode-terminal": "^0.12.0"
  }
}
"""

WHATSAPP_BRIDGE_JS = r"""#!/usr/bin/env node
import express from 'express';
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

mkdirSync(SESSION_DIR, { recursive: true });

const logger = pino({ level: 'warn' });
const app = express();
app.use(express.json({ limit: '2mb' }));

let sock = null;
let connectionState = 'disconnected';

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
