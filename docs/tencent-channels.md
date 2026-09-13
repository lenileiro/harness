# Weixin personal accounts and QQ Bot

Both adapters use Harness's durable channel inbox/outbox and gateway approval
boundary. These implementations were checked against pinned Hermes commit
[`939e45c91d751fadd94dcd1b873ac3cb44846213`](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213/gateway/platforms),
Tencent's published Weixin plugin source, and Tencent's QQ protocol documentation.
All verification used fake APIs, local cryptography, and loopback sockets. No real
Tencent account was paired and no external message was sent.

## Weixin setup

Run `harness channel weixin-pair --cwd /path/to/workspace`. Scan the displayed QR
with the intended Weixin account and supply a verification code if Tencent asks
for one. The explicit pairing flow has a bounded timeout, stops on cancellation,
and saves credentials to `.harness/channels/weixin-account.json` with mode `0600`.
It prints a configuration snippet containing the file path and paired owner ID;
the token is never printed. Existing credentials require explicit `--replace`.

```toml
[channels.weixin]
account_file = "/path/to/workspace/.harness/channels/weixin-account.json"
allowed_users = ["OWNER_ID_FROM_PAIRING"]
```

Start `harness channel run weixin --cwd /path/to/workspace`. Inspect queues with
`harness channel status weixin --cwd /path/to/workspace`.

Alternatively, set `token_env` to an environment variable containing the pairing
JSON. Existing raw iLink tokens are supported with `token_env` and explicit
`app_id = "PAIRED_BOT_ID"`. An environment token takes precedence over
`account_file`; the file is read only when explicitly configured and bounded to
16 KiB. Account IDs must agree when both JSON credentials and `app_id` are set.
Changing the paired bot requires separate channel state; use a new workspace or
profile when connecting another account.

Long polling persists accepted messages and peer reply context before advancing
the iLink cursor. Restarting resumes the cursor and retains the latest sequence's
reply token for each account/peer. Duplicate message IDs are not dispatched again;
identical text with a new ID remains a new message. Reply tokens never enter the
model transcript. Sending requires a prior authenticated message from that peer,
and an expired session never causes a retry without its scoped token.

Text and encrypted image/file/video transfer are implemented with a 20 MiB media
bound. AES-128-ECB with PKCS#7 follows Tencent's wire format; files with supplied
MD5 metadata are checked after decryption. Downloads use the official HTTPS CDN
without redirects or bot authorization headers. Native voice input retains SILK
bytes and any transcript supplied by iLink; a model must support the audio format.
Arbitrary outbound audio is delivered as a file, without claiming SILK conversion.
Native voice encoding, typing indicators, group routing, and cross-region pairing
redirects outside the configured official iLink origin are not implemented. Group
mode fails explicitly. Pairing sessions that expire or are already bound require
an explicit restart instead of background account changes.

The primary protocol reference is Tencent's
[`@tencent-weixin/openclaw-weixin` package](https://www.npmjs.com/package/@tencent-weixin/openclaw-weixin),
inspected at version **2.4.8** (source only). Its API types, QR flow and CDN modules
define the implemented requests. This is a personal-account iLink integration;
enterprise WeCom remains a separate channel.

## QQ Bot setup

Create a QQ Bot application and enable the message capabilities required by its
deployment. Set its client secret in an environment variable:

```toml
[channels.qqbot]
app_id = "123456789"
token_env = "QQ_CLIENT_SECRET"
receive_mode = "websocket"
allowed_users = ["123456789:c2c:USER_OPENID"]
allowed_channels = ["123456789:c2c:USER_OPENID"]
```

Run `harness channel run qqbot --cwd /path/to/workspace`. Tokens are acquired using
the official [AppID/client-secret exchange](https://github.com/tencent-connect/bot-docs/blob/master/docs/develop/api-v2/dev-prepare/interface-framework/api-use.md)
and refreshed with a single request when concurrent callers need renewal.

The default native [WebSocket Gateway](https://github.com/tencent-connect/bot-docs/blob/master/docs/develop/api-v2/dev-prepare/interface-framework/reference.md)
persists its session/sequence after message ingestion, resumes after reconnects,
handles invalid sessions, and closes the owned socket on missed heartbeat
acknowledgments or cancellation. The adapter requests the C2C/group, public guild
mention, and guild direct-message intents. Application eligibility and current
Gateway availability must be verified with the selected live bot account.

Alternatively, use `receive_mode = "webhook"`, set `webhook_path`, `listen_host`
and `listen_port`, and configure the public HTTPS callback in QQ's portal. Harness
answers the signed validation challenge; other events require the raw-body
[Ed25519 signature](https://github.com/tencent-connect/bot-docs/blob/master/docs/develop/api-v2/dev-prepare/interface-framework/sign.md)
and a timestamp within five minutes. A validation challenge does not enqueue model
work. Tests include Tencent's published seed/public-key vector.

User and conversation identities include the application and destination type:

| Destination | `allowed_users` | `allowed_channels` |
| --- | --- | --- |
| C2C | `APP_ID:c2c:USER_OPENID` | `APP_ID:c2c:USER_OPENID` |
| Group | `APP_ID:group:MEMBER_OPENID` | `APP_ID:group:GROUP_OPENID` |
| Guild channel | `APP_ID:guild:USER_ID` | `APP_ID:guild:CHANNEL_ID` |
| Guild direct message | `APP_ID:guild:USER_ID` | `APP_ID:dm:GUILD_DM_ID` |

Enable `allow_groups` for group and guild-channel events. The default mention
requirement accepts native at-message events; guild events without a mention are
filtered. Sender, group and DM routes are distinct and never inferred from a bare
target ID.

Replies bind to an authenticated incoming message with a five-minute window.
C2C/group deliveries persist `msg_id` plus a monotonically allocated `msg_seq` so
explicit retries retain the same deduplication identity. Guild/DM endpoints use
their native `msg_id` request shape and do not claim server-side sequence
deduplication. Expired old replies remain expired; a new incoming message creates
a fresh reply opportunity. Proactive delivery outside this window is not enabled.

Media downloads have a 20 MiB aggregate bound, allow only platform hosts, and reject
redirects. C2C/group uploads send inline bytes with `srv_send_msg = false` before a
separate acknowledged media message. Guild/DM native uploads currently support
images only. Buttons, keyboards, reactions, editing, audio transcoding and QQ's
scan-to-create-app onboarding remain separate from this transport implementation.
Use QQ's developer portal for app creation and its live capability checks.
