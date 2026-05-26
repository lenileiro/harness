# WhatsApp QR Pairing Plan

This repo now targets **personal-number QR pairing** for WhatsApp instead of the
Meta Cloud API transport.

## Goals

- use a local WhatsApp Web session
- pair through a CLI-driven QR flow
- keep session state under the workspace, not in cloud credentials
- send outbound notifications and replies through a local bridge
- avoid marking WhatsApp enabled until pairing succeeds

## Architecture

1. `harness gateway whatsapp setup`
   - configures mode and allowlist
   - materializes a local Node bridge bundle under `.harness/gateway/whatsapp/bridge`
   - optionally installs bridge dependencies
   - optionally runs QR pairing

2. `harness gateway whatsapp pair`
   - runs the bridge in `--pair-only` mode
   - waits for QR scan completion
   - enables WhatsApp only after `creds.json` exists

3. `harness gateway whatsapp start`
   - starts the long-lived local bridge
   - bridge owns the Baileys session and localhost HTTP endpoints

4. `harness gateway whatsapp send`
   - posts outbound messages to the local bridge

## Storage

- config: `.harness/gateway/whatsapp/config.json`
- bridge bundle: `.harness/gateway/whatsapp/bridge/`
- session: `.harness/gateway/whatsapp/session/`
- bridge log: `.harness/gateway/whatsapp/bridge.log`

## Scope

This slice implements:

- CLI setup/status/pair/start/send
- local bridge asset generation
- local send/health transport
- scheduler notification hook integration

This slice intentionally does **not** yet implement:

- inbound message polling and automatic gateway dispatch from the bridge
- advanced media flows
- multi-account WhatsApp sessions
