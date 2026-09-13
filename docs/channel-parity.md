# Messaging channels and remaining reference coverage

Harness now provides Telegram long polling, Discord Gateway, Slack Socket Mode,
Signal through a local `signal-cli` process, email over IMAP/SMTP, and the existing
WhatsApp QR bridge. Matrix client sync and Google Chat/Feishu/Lark authenticated
webhooks, Teams Bot Connector, and DingTalk Stream also feed the scoped gateway and durable local
inbox/outbox. Mattermost, IRC, ntfy, LINE, SMS/Twilio and WeCom internal-app
workflows are described in [their setup guide](community-and-business-channels.md).
SimpleX, Buzz, Raft, Photon and Yuanbao have [a separate setup guide](local-client-and-yuanbao-channels.md).
WhatsApp Cloud, BlueBubbles and Graph notifications are covered in
[Apple and cloud channels](apple-and-cloud-channels.md); Weixin and QQBot in
[Tencent channels](tencent-channels.md). Native Google Chat/Feishu/Teams transfers
are described in [the attachment guide](native-channel-attachments.md); WeCom
smart bots in [their WebSocket guide](wecom-bot.md).
External accounts have not been exercised; automated tests use fake
HTTP services, actual loopback WebSocket servers, and a real local fake Signal
subprocess. WhatsApp media helpers run under Node with fake Baileys transport.

## Configuration and commands

Configure environment **variable names**, never tokens, in the selected profile's
TOML configuration:

```toml
[channels.telegram]
token_env = "TELEGRAM_BOT_TOKEN"
allowed_users = ["123456789"]
allow_groups = false
require_mention = true

[channels.discord]
token_env = "DISCORD_BOT_TOKEN"
allowed_users = ["123456789012345678"]
allowed_channels = ["234567890123456789"]
allow_groups = true
require_mention = true

[channels.slack]
token_env = "SLACK_BOT_TOKEN"
app_token_env = "SLACK_APP_TOKEN"
allowed_users = ["T_WORKSPACE:U_USER"]
allowed_channels = ["T_WORKSPACE:D_DM_CHANNEL"]

[channels.signal]
token_env = "SIGNAL_ACCOUNT"
signal_command = "signal-cli"
allowed_users = ["+15551234567"]

[channels.email]
token_env = "EMAIL_PASSWORD"
username = "bot@example.org"
imap_host = "imap.example.org"
smtp_host = "smtp.example.org"
imap_port = 993
smtp_port = 465
mailbox = "INBOX"
allowed_users = ["owner@example.org"]

[channels.matrix]
token_env = "MATRIX_ACCESS_TOKEN"
homeserver = "https://matrix.example.org"
allowed_users = ["@owner:matrix.example.org"]
allowed_channels = ["!room:matrix.example.org"]
allow_groups = true
require_mention = true

[channels.google_chat]
token_env = "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON"
audience = "https://bot.example.org/google-events"
app_id = "users/BOT_USER_ID"
webhook_path = "/google-events"
listen_port = 8766
allowed_users = ["users/USER_ID"]
allowed_channels = ["spaces/SPACE_ID"]
allow_groups = true

[channels.teams]
token_env = "TEAMS_APP_SECRET"
app_id = "APP_ID"
tenant_id = "TENANT_ID"
webhook_path = "/teams-events"
listen_port = 8768
allowed_users = ["TENANT_ID:AAD_OBJECT_ID"]
allow_groups = true

[channels.dingtalk]
token_env = "DINGTALK_APP_SECRET"
app_id = "APP_KEY"
allowed_users = ["CORP_ID:STAFF_ID"]
allow_groups = true

[channels.feishu]
token_env = "FEISHU_APP_SECRET"
app_token_env = "FEISHU_VERIFICATION_TOKEN"
signing_secret_env = "FEISHU_ENCRYPT_KEY"
app_id = "APP_ID"
webhook_path = "/feishu-events"
listen_port = 8767
allowed_users = ["TENANT_KEY:OPEN_USER_ID"]
allowed_channels = ["CHAT_ID"]
allow_groups = true
```

Set credentials in the environment or the selected profile's credential store,
then run one foreground process for each desired channel:

```sh
harness channel run telegram --cwd /path/to/workspace
harness channel run discord --cwd /path/to/workspace
harness channel run slack --cwd /path/to/workspace
harness channel run signal --cwd /path/to/workspace
harness channel run email --cwd /path/to/workspace
harness channel run matrix --cwd /path/to/workspace
harness channel run google_chat --cwd /path/to/workspace
harness channel run feishu --cwd /path/to/workspace
harness channel run teams --cwd /path/to/workspace
harness channel run dingtalk --cwd /path/to/workspace
harness channel status telegram --cwd /path/to/workspace
harness channel retry telegram DELIVERY_ID --cwd /path/to/workspace
```

`--allow-user` can supply an explicit allowlist at launch. Empty user allowlists
are rejected. `harness --profile NAME channel run ...` uses that profile for the
whole process. Optional `channels.NAME.profile` must match; identities are never
switched by changing environment variables during a message. One runner may hold
a given workspace/channel lock. Replacing bot credentials with a different bot
requires a separate workspace/profile, so old cursors and destinations cannot
silently transfer to another identity.

Telegram needs a Bot API token and no active webhook. Group reception follows
Telegram's bot privacy settings; mentions and replies to the bot are recognized.
Discord needs an installed bot with read/send permissions. The privileged
`MESSAGE_CONTENT` intent is requested only for group operation without a mention
requirement; enable that intent in the developer portal for that configuration.
Slack needs Socket Mode, an app token with `connections:write`, a bot token with
`chat:write`, relevant message/app-mention event subscriptions, and channel
membership. File transfer additionally needs `files:read`/`files:write`.
Signal needs an already registered or linked `signal-cli` account; `SIGNAL_ACCOUNT`
contains its account number. Harness owns the JSON-RPC child and terminates it
when its runner stops.

Matrix requires an existing bot access token and joined rooms. Its initial sync
establishes a cursor without dispatching historic messages; subsequent events
are ingested before advancing the cursor. Direct-room account data plus member
counts identify a two-member DM; other rooms use the group policy. Opt-in Matrix
E2EE uses matrix-nio 0.26.0 and vodozemac, explicit device-fingerprint pins,
private per-profile/device stores and encrypted media. It rejects plaintext
fallback and unverified senders/devices, and retains undecrypted ciphertext for
later keys across restarts. Without E2EE enabled, encrypted rooms remain blocked
for outgoing plaintext. Text transactions retain their delivery ID. See
[Matrix encryption setup](../packages/cli/MATRIX_E2EE.md); cross-signing and
server-side key backup are not supported by this SDK.

Google Chat requires a configured HTTP interaction app, a service account with
Chat API access, and the public HTTPS endpoint URL as its authentication audience.
`token_env` contains the service-account JSON itself, not a local filename.
The adapter exchanges signed service-account assertions for renewable `chat.bot`
access tokens. Incoming ID tokens must have Google's RS256 signature, the exact
audience, current expiry, and a verified `chat@system.gserviceaccount.com` identity.
Project-number audience JWTs and Workspace add-on event schemas are not currently
implemented. `app_id` identifies the bot's `users/...` resource for exact mentions.

Feishu requires the app secret, event verification token, and Encrypt Key. Its
encrypted URL challenge is answered without dispatching a run. Normal callbacks
require a fresh signed body, the verification token, and matching app ID. The
adapter renews tenant access tokens and replies to native message threads.
Use `[channels.lark]` and `channel run lark` for the same protocol on the official
Lark API domain. Subscribe to `im.message.receive_v1` and grant the bot message
permissions. Both transports support native attachments with account-specific
upload permissions and format limits documented in the
[attachment guide](native-channel-attachments.md). Feishu/Lark native approval cards are described in the [decision-button guide](channel-approval-cards.md);
other interactive actions remain separate work.

Webhook runners bind `127.0.0.1` by default. Put the configured path behind an
operator-managed HTTPS reverse proxy, preserving authorization/signature headers
and the raw body; no public tunnel or account is provisioned by Harness. Configure
different `listen_port` values for concurrent webhook channels. The listener
limits event bodies to 1 MiB, bounds concurrent requests, and acknowledges only
after authentication and durable ingestion. Automated tests send real signed
requests to an ephemeral loopback listener; production TLS/proxy/account setup
has not been exercised.

The implementations follow the primary [Telegram Bot API](https://core.telegram.org/bots/api),
[Discord Gateway](https://docs.discord.com/developers/events/gateway),
[Slack Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/),
and [signal-cli JSON-RPC](https://github.com/AsamK/signal-cli/blob/master/man/signal-cli-jsonrpc.5.adoc)
contracts. Slack uploads use the current
[external upload flow](https://docs.slack.dev/reference/methods/files.getUploadURLExternal/).
Additional contracts: [Matrix client-server API](https://spec.matrix.org/latest/client-server-api/),
[Google Chat request verification](https://developers.google.com/workspace/chat/verify-requests-from-chat),
[Google Chat message creation](https://developers.google.com/workspace/chat/api/reference/rest/v1/spaces.messages/create),
and the official [Feishu SDK callback verifier](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/event/dispatcher_handler.py).

## Identity, approvals, and delivery

Transport-authenticated user IDs and conversation IDs feed the existing gateway
bindings. Telegram uses `CHAT_ID` or `CHAT_ID:TOPIC_ID`; Discord uses its channel or
thread snowflake. Slack user IDs are `TEAM_ID:USER_ID` and conversations are
`TEAM_ID:CHANNEL_ID:ROOT_TIMESTAMP`; direct messages have an empty timestamp.
Signal uses the sender number/UUID and either that sender or `group:GROUP_ID`.
Email uses the confirmed mailbox address as the user and conversation ID.
Matrix uses exact Matrix user IDs; its conversation ID is compact JSON
`["ROOM_ID","THREAD_ROOT_EVENT_ID"]`, with an empty root for unthreaded chat.
Google Chat uses `users/USER_ID` and compact JSON `["spaces/SPACE_ID","THREAD_NAME"]`;
DMs have an empty thread name. Feishu/Lark users are `TENANT_KEY:OPEN_USER_ID` and
conversations are compact JSON `["CHAT_ID","ROOT_MESSAGE_ID"]`, empty-root for DMs.
These are also the destinations used by scheduled jobs; keep the JSON exact.

Email `From` headers do not authenticate their sender. The first allowlisted
email therefore triggers only a challenge delivered to that actual mailbox.
A later email must reply to a random Message-ID Harness sent to the same mailbox
before it can dispatch a model turn or approve an action. These reply capabilities
expire after 24 hours and must be kept private. Challenge messages are limited
to once per hour per mailbox. TLS protects both authenticated IMAP and SMTP
connections; passwords are read from the configured environment reference.

Ordinary channel users can chat and use `status`, `runs`, `approvals`, `approve ID`,
and `deny ID`. Other legacy workspace controls require an additional exact user
entry in `operator_users`. Those controls remain privileged host operations.
New conversational sessions retain the gateway's scoped memory, private-file
boundary, queued mutation approvals, expiration and replay-claim rules. Browser
and execution backend configurations are rejected for remote sessions until
an isolated identity runtime exists.

Every gateway transport also accepts these explicit conversation commands:

- `/model` shows the active provider/model; `/model MODEL_ID` selects a model on
  that provider. Existing context is copied into a new owner-bound session while
  the original session and its execution settings remain available.
- `/retry` resubmits the last stored user prompt and its attachments through the
  ordinary scoped runtime and queued approval policy.
- `/undo` archives and removes the last user turn and clears its short continuity
  summary. It affects the transcript; workspace changes remain.
- `/compress` summarizes older owned context, archives the original transcript,
  and keeps the original unchanged if summarization fails or does not reduce the
  token count. Codex summaries use an isolated app-server session without tools;
  explicit exec-mode configuration is rejected for this remote operation.
- `/usage` reports token counts from the current session's usage events, with an
  explicit truncation flag if the 100,000-record reporting bound is reached.

Commands use the same conversation lock and exact transport/user/thread binding.
Transcript/model changes and retry refuse active, paused, pending-approval, or
unfinished granted-action sessions. These are typed slash messages on the current
channels. Telegram, Discord, Slack, Feishu/Lark and Teams also provide
[native approval buttons](channel-approval-cards.md), backed by the same persisted
requests. Telegram, Discord, Slack and Google Chat can render optional
[clarification choices](clarification.md) when that workflow is explicitly enabled;
agents gather context autonomously by default. Six transports also show
[bounded run feedback](channel-progress.md) and replace the owned status message
with the final response. Native command menus and general interactive forms remain separate.

Telegram cursors advance only after durable ingestion. Discord stores its Gateway
session/sequence for resume and maintains independent heartbeat handling. Slack
acknowledges envelopes after durable ingestion, before model execution. Duplicate
platform events cannot dispatch the same saved inbox item again. Email tracks
UIDVALIDITY/UID and deduplicates message identities across mailbox reindexing.

Replies and media become independently stored delivery items. Text is split at
platform limits without breaking Unicode characters. Rate-limit delays are
persisted; sent chunks are not repeated when later chunks are retried. Scheduled
prompt/reminder notifications transfer from the scheduler outbox into this channel
outbox with the original destination and an idempotent source ID. Keep the channel
runner active to deliver them.

A crash during model dispatch or a possibly successful network send leaves an
`uncertain` item. Inspect the original session or destination before explicit
`channel retry`; `--inbound` retries an uncertain dispatch. Arbitrary messaging
APIs do not provide an exactly-once delivery guarantee. Status output reports
IDs and states without private message bodies or credentials.

Teams uses a single-tenant Microsoft app with the Teams channel enabled and a
public HTTPS reverse proxy to its callback listener. Bot Connector JWTs must
match the configured app and tenant, the exact signed service URL, and a signing
key endorsed for Teams. Only public-cloud Connector destinations are supported.
Replies reuse an authenticated persisted conversation route; channel thread roots
remain stable while personal/group chats retain one conversation. User IDs are
`TENANT_ID:AAD_OBJECT_ID` (falling back to the connector sender ID), and channel
allowlists use `TENANT_ID:CONVERSATION_ID`. Native inline images and personal-chat
file consent are described in the [attachment guide](native-channel-attachments.md);
group file uploads require separate Graph permissions and are unsupported.
The protocol follows [Microsoft Bot Connector authentication](https://learn.microsoft.com/en-us/azure/bot-service/rest-api/bot-framework-rest-connector-authentication).

DingTalk requires an internal robot app with Stream mode enabled. It opens the
[official authenticated Stream protocol](https://opensource.dingtalk.com/developerpedia/docs/learn/stream/protocol/)
with the app secret and persists messages before acknowledging callback frames.
User allowlists use `CORP_ID:STAFF_ID` (falling back to sender ID); conversation
allowlists use `CORP_ID:CONVERSATION_ID`. Text replies use the private session
webhook received from that authenticated stream, restricted to the official
DingTalk session endpoint. A reply after its expiry requires a new inbound message
and explicit delivery retry. Native image/file download codes are exchanged with
the authenticated app; bounded downloads permit only DingTalk/Alibaba media
hosts. Rich-text images and recognized voice text are accepted. Outbound images
may reference an explicitly supplied HTTPS image URL in native Markdown; inline
files and unattended proactive delivery remain unsupported, matching the pinned
Stream adapter's media scope. Group messages follow `isInAtList` and the mention policy.

## Media boundary

Image/audio/file attachments use the versioned `MediaAttachment` schema and survive
session and outbox persistence. Telegram, Discord and Slack downloads are limited
to their authenticated platform hosts, with no redirects and a 20 MiB aggregate
input bound. Signal obtains attachment data through its own JSON-RPC process.
Email decodes bounded MIME attachments. Remote/local paths in attachment names
never become filesystem reads. Outbound uploads require inline bytes; arbitrary
remote media URLs are rejected by those transports. DingTalk's URL-only Markdown
image send is an explicit exception; Harness forwards its HTTPS URL without
fetching it. A provider must support the requested media type;
transport support alone does not imply model support.
Matrix downloads MXC resources only through its authenticated homeserver and
uses native uploads with the same input-byte bound; arbitrary URLs are rejected.
In E2EE mode it encrypts bytes before upload and verifies encrypted attachment
integrity before exposing downloaded content.

The WhatsApp bridge downloads supported media as bounded streams after its
allowlist check and passes base64 via child stdin. It sends image/audio/document
buffers and splits long text replies. View-once messages are not copied into
durable media artifacts. WhatsApp's older bridge queue/replay behavior is distinct
from the new channel inbox/outbox; live Baileys compatibility remains unverified.

## Exact pinned reference inventory

The parity reference remains Nous Research Hermes commit
[`939e45c91d751fadd94dcd1b873ac3cb44846213`](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213).
Its [bundled platform plugins](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213/plugins/platforms)
are shipped in the official repository. Being implemented as a plugin does not
make these third-party-only features or remove them from the reference backlog.
The [gateway platform directory](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213/gateway/platforms)
contains additional adapters and infrastructure.

| Reference surface | Harness status |
| --- | --- |
| Telegram, Discord, Slack | Text/media workflows, owner-bound native approval buttons, run feedback and optional question choices implemented; live account verification pending. |
| Signal, email | Local Signal RPC and TLS mail workflows implemented; live account verification pending. |
| WhatsApp | Existing QR bridge plus native media transfer; separate bridge recovery limits above. |
| Matrix | Persistent text/media sync plus opt-in SDK E2EE, exact device-fingerprint trust, private key stores, encrypted media and restart recovery; cross-signing/server key backup remain unsupported. |
| Google Chat | Authenticated text, native attachments, owned run feedback and optional question choices; uploads require user OAuth. Native approval cards are not implemented. See the [attachment guide](native-channel-attachments.md). |
| Feishu/Lark | Authenticated text/media and native approval cards; native audio retains format limits. Other interactions remain separate. |
| Teams, DingTalk | Text and native inbound attachments implemented. Teams supports native approval cards, inline images and personal file consent; DingTalk sends hosted-image Markdown. Tenant, file-permission and expiring-reply restrictions apply above. |
| Mattermost, IRC, ntfy, LINE, SMS, WeCom | Text workflows plus native Mattermost/LINE/WeCom smart-bot media; see the [setup guide](community-and-business-channels.md) and [smart-bot guide](wecom-bot.md). IRC is account-authenticated channel-only; ntfy uses signed owner envelopes. SMS/ntfy media is absent from the pinned reference too. |
| Bundled plugins: buzz, photon, raft, simplex | Working text clients described in [the setup guide](local-client-and-yuanbao-channels.md). Photon includes device setup; Raft wakes are metadata-only with local results and approved native CLI actions. Per-platform media, reactions and other limitations remain explicit. |
| Bundled plugins: a2a, homeassistant | Agent protocol and approved home automation interfaces are tracked in [feature parity](feature-parity.md), with separate identity and side-effect boundaries. |
| Gateway adapters: bluebubbles, msgraph_webhook, whatsapp_cloud | Implemented with [documented contracts](apple-and-cloud-channels.md). Graph is notification ingress with local durable replies, not a general Graph mail sender. |
| Gateway adapters: weixin, qqbot | Implemented with [native authentication and media](tencent-channels.md); Weixin includes QR pairing. Region/group/codec and interactive-control limitations remain explicit. |
| Gateway adapter: yuanbao | HTTPS signing, authenticated binary WebSocket, ACK-after-persistence and correlated text delivery implemented. Native media and richer message behaviors remain incomplete. |
| Gateway infrastructure: api_server, webhook, event and API room/run helpers | Separate serving/event capabilities, not additional messaging accounts; compare against Harness serving interfaces individually. |

Platform features beyond the implementations listed above include additional
OAuth installation flows, reactions, arbitrary message editing, native command
menus and richer group policies. Live provider/channel conformance also requires
separate evidence. This inventory does not claim full Hermes parity.
