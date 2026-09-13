# Mattermost, IRC, ntfy, LINE, SMS and WeCom

These adapters use the shared scoped gateway, queued approvals, conversation
commands and durable channel inbox/outbox. Start them with `harness channel run
NAME --cwd PATH`, or use `harness channel run-all` for the configured profile.
All credentials below are environment variable references. Protocol tests use
fake accounts and actual local HTTP, WebSocket and TLS servers; live account
conformance remains unverified.

## Mattermost

```toml
[channels.mattermost]
homeserver = "https://mattermost.example.org"
token_env = "MATTERMOST_BOT_TOKEN"
allowed_users = ["USER_ID"]
allowed_channels = ["CHANNEL_ID"]
allow_groups = true
require_mention = true
```

Create a bot account/token and grant access to each configured channel. Explicit
channel IDs are required for bounded REST history recovery. The adapter verifies
the bot and channels, authenticates the native WebSocket before dispatch, and
uses immutable post/user IDs. Group replies preserve the original root post;
DMs retain one conversation. Mention matching uses the verified bot username.

The first connection establishes a server-timestamp baseline; historic messages
are not executed. Reconnects page backward through up to 10,000 posts per channel,
then ingest unseen posts in order before advancing the saved timestamp. An
unrecoverable history gap preserves the cursor and reports failure. Live and
recovered copies share the same deduplication key. Deleted, system, webhook and
self-bot posts are ignored. Text replies use the native create-post endpoint. Inbound files are downloaded
only after their file metadata matches the authenticated post. Outbound inline
attachments use native multipart upload and retain the exact channel/root post.
Transfers are bounded to 20 MiB; native interactive actions remain unsupported. See the official
[WebSocket API](https://docs.mattermost.com/api/reference/connect-web-socket) and
[channel history API](https://docs.mattermost.com/api/reference/get-posts-for-channel).

## IRC over TLS

```toml
[channels.irc]
homeserver = "ircs://irc.example.org:6697"
username = "registered-bot-account"
app_id = "harnessbot"
token_env = "IRC_SASL_PASSWORD"
allowed_users = ["irc.example.org:6697:registered-owner-account"]
allowed_channels = ["#private-project"]
allow_groups = true
require_mention = true
```

The server must support TLS with a valid certificate, SASL PLAIN, `account-tag`,
`server-time`, and `message-tags`. Harness completes SASL and confirms channel
joins before accepting deliveries. User ownership comes from the server's
account tag, using IRC case folding, not a reusable nickname or client-only tag.
Replies go only to the explicitly configured channels. Nickname DMs are disabled
because the recipient could change while an approval or model response waits.
Channel members can see messages sent to that channel; choose channel membership
accordingly.

Message IDs persist across reconnects. A server timestamp plus the complete
original frame provides conservative deduplication when `msgid` is unavailable.
Harness does not request IRC history; messages missed while disconnected require
server/bouncer replay. Successful outbound delivery means a bounded frame was
written to the TLS connection, not a server-confirmed read receipt. Text is
paced and bounded below IRC's byte limit; control bytes cannot inject commands.
Media, DMs and native history requests remain unsupported. The pinned reference
also lacks native history requests and attachments. Its nickname DM support is
excluded here because IRC account tags authenticate inbound users without
providing account-bound outbound addressing. Protocol references:
[IRCv3 SASL](https://ircv3.net/specs/extensions/sasl-3.1.html),
[account tags](https://ircv3.net/specs/extensions/account-tag.html), and
[message tags](https://ircv3.net/specs/extensions/message-tags.html).

## ntfy

```toml
[channels.ntfy]
homeserver = "https://ntfy.example.org"
token_env = "NTFY_ACCESS_TOKEN"
app_token_env = "NTFY_OWNER_SIGNING_SECRET"
username = "owner"
allowed_users = ["owner"]
topic = "private-harness-input"
reply_topic = "private-harness-output"
```

Use two protected topics and a signing secret of at least 32 bytes. The service
access token authorizes read/write API access; topic ACLs must restrict readers
of private output. ntfy messages do not expose a trustworthy publisher identity,
so ordinary unsigned bodies cannot become agent prompts or approval decisions.
A separate HMAC-SHA256 envelope authenticates the single configured owner, input
topic, message nonce, timestamp and text. Envelopes expire after 24 hours and
reuse of a nonce is deduplicated even across different ntfy message IDs.

Publish a signed prompt with the same profile configuration:

```sh
harness channel publish-ntfy "Summarize the current project status"
harness channel publish-ntfy "approve APPROVAL_ID"
```

The command outputs only its message ID. Input envelopes fit ntfy's 4096-byte
message limit. Replies publish ordinary text to the separate output topic, so
they cannot loop back as commands. Polling saves a cursor after ingestion; first
startup establishes a cache baseline without executing old messages. Explicitly
truncated cache replays preserve the cursor and fail instead of claiming complete
recovery. Server cache retention can still limit recovery after long downtime.
Attachments and arbitrary multi-publisher identities are not implemented. The
pinned reference adapter is text-only too; ntfy attachment features are outside
this parity scope. See
[ntfy subscription API](https://docs.ntfy.sh/subscribe/api/) and
[JSON publishing](https://docs.ntfy.sh/publish/#publish-as-json).

## LINE

```toml
[channels.line]
token_env = "LINE_CHANNEL_ACCESS_TOKEN"
app_token_env = "LINE_CHANNEL_SECRET"
webhook_path = "/line-events"
# Required only for outbound native image/audio/video hosting:
webhook_url = "https://bot.example.org/line-events"
listen_port = 8770
allowed_users = ["U_USER_ID"]
allow_groups = false
```

Configure a LINE Messaging API bot and route its public HTTPS callback URL to
this local listener. Signatures are verified over the original body before JSON
parsing, and the callback destination must match the authenticated bot. Native
mention offsets use UTF-16 units, including emoji. User IDs scope approvals;
conversation IDs retain user/group/room type. Groups require `allow_groups` and
the mention policy, and can be restricted with `allowed_channels` containing
LINE group/room IDs.

Replies use push messages with a stable `X-Line-Retry-Key`. A recognized already
accepted request does not resend the message. Messaging quotas and recipient
access apply. Inbound image, audio, video and file events download from LINE
after authentication and allowlist checks. Outbound inline JPEG/PNG images,
MP3/MP4 audio and MP4 video use LINE native messages. Audio requires actual
`duration_ms` metadata. The exact public HTTPS `webhook_url` must reach this
listener; it serves only queued attachment bytes under unguessable links that
expire after one hour. The private cache is bounded to 100 MiB/100 items and
prunes expired entries on publication. No local filesystem path is served.
General outbound documents are unsupported by the reference LINE adapter too.
Native cards, postback actions and command installation remain unsupported. See LINE's
[signature verification](https://developers.line.biz/en/docs/messaging-api/verify-webhook-signature/)
and [retry guidance](https://developers.line.biz/en/docs/messaging-api/retrying-api-request/).

## SMS through Twilio

```toml
[channels.sms]
app_id = "AC_ACCOUNT_SID"
username = "+15550000001"
token_env = "TWILIO_AUTH_TOKEN"
webhook_url = "https://bot.example.org/sms-events"
webhook_path = "/sms-events"
listen_port = 8771
allowed_users = ["+15550000002"]
```

Configure that Twilio number's incoming-message URL exactly as `webhook_url`.
The signature binds all form parameters and the configured external URL;
untrusted proxy headers cannot change it. Duplicate form parameters, different
account SIDs and messages addressed to another receiving number are rejected.
Each conversation retains the bot's number and the exact allowed sender number.
Incoming Message SIDs deduplicate retries, and empty TwiML acknowledges only
after durable ingestion. Replies use authenticated Twilio REST requests.

Carrier segmentation, cost and delivery receipts are outside Harness's local
outbox acknowledgment. Uncertain sends require destination inspection before
manual retry. Number ownership follows the SMS carrier channel; additional
out-of-band account authentication is not provided. MMS, group SMS and delivery
receipt tracking are unsupported. The pinned SMS adapter is text-only; MMS is a
vendor capability outside the pinned parity scope. Protocol references: Twilio
[request authentication](https://www.twilio.com/docs/usage/security) and
[API requests](https://www.twilio.com/docs/usage/requests-to-twilio).

## WeCom internal application

```toml
[channels.wecom]
tenant_id = "CORPORATION_ID"
app_id = "1000001"
token_env = "WECOM_APP_SECRET"
app_token_env = "WECOM_CALLBACK_TOKEN"
signing_secret_env = "WECOM_ENCODING_AES_KEY"
webhook_path = "/wecom-events"
listen_port = 8772
allowed_users = ["CORPORATION_ID:USER_ID"]
```

Configure an internal application with message reception enabled, the callback
token and its 43-character EncodingAESKey. A public HTTPS reverse proxy exposes
GET verification and POST XML callbacks on the same listener. Harness verifies
the callback signature and freshness before AES decryption, then validates the
ciphertext framing, corporation ID and application ID. XML entity declarations
and duplicate fields are rejected. Token renewal and explicit expired-token
responses are handled before retrying an unsent application message.

Only individual allowed users receive text messages; broadcast expressions such
as `@all` and multi-recipient separators are rejected. Message IDs deduplicate
callbacks, while conversation IDs bind corporation, application and user.
This default `wecom_mode = "internal_app"` supports individual text workflows.
The separate [native smart-bot mode](wecom-bot.md) supports WebSocket callbacks,
group conversations and media. Cards and internal-app media remain unsupported. Implementation was
checked against the [WeCom team's protocol library](https://github.com/sbzhu/weworkapi_python),
including callback framing and application-token renewal; the official web
reference was unavailable to the research client during implementation.
