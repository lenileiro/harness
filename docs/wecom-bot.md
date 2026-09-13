# WeCom native smart bots

WeCom smart-bot mode uses the official authenticated WebSocket at
`wss://openws.work.weixin.qq.com`. It is separate from the default internal-app
XML webhook configuration. Create a long-connection smart bot in WeCom, copy its
bot ID, and store its secret in the named environment variable:

```toml
[channels.wecom]
wecom_mode = "bot"
app_id = "BOT_ID"
token_env = "WECOM_BOT_SECRET"
allowed_users = ["USER_ID"]
allow_groups = true
require_mention = true
username = "Harness"
# Optional restriction, using native group chat IDs:
# allowed_channels = ["group:CHAT_ID"]
```

Run `harness channel run wecom --cwd /path/to/workspace`. No inbound HTTP port or
public tunnel is needed. Group mention matching requires the bot's exact display
name in `username`; alternatively disable `require_mention` explicitly. Messages
must match the subscribed bot ID and configured user/channel allowlists. Users
are native WeCom user IDs. Conversations are compact JSON
`["BOT_ID","dm|group","TARGET_ID","USER_ID"]`; a group callback route belongs to
the requesting user. Switching an existing profile from internal-app mode to a
different smart-bot identity is rejected; use another profile/workspace.

Harness verifies subscription acknowledgement before receiving prompts, runs
heartbeats independently, persists message IDs before dispatch, and correlates
outbound responses with native request acknowledgements. Duplicate events do
not refresh or replace an existing reply capability. A server revocation caused
by a competing connection stops reconnection until the operator inspects it.
The server limits one active connection per bot; this adapter does not provision
bots or perform WeCom QR onboarding.

Direct replies use native proactive messages. Group replies use the requesting
user's callback, with a five-minute local lifetime; split text updates one stream
and finishes its last part, with a 20 KiB total stream bound. Expired group
replies fail visibly and require a new message from that owner. Uncertain sends
retain the ordinary channel outbox inspection/retry requirement. Native account
permissions and response-window limits still apply.

Inbound text, recognized voice text, images, files, videos, mixed messages and
quoted content are supported. Native encrypted media downloads use HTTPS from
WeCom/Tencent hosts, AES-256-CBC and strict padding validation. The queue downloads
at most 16 attachments and 20 MiB in aggregate. Native attachment URLs can expire
in approximately five minutes, so long queue delays can prevent downloads.
Voice recognition text does not imply access to the original audio bytes.

Outbound inline attachments use native upload-init, acknowledged 512 KiB chunks,
upload-finish and message delivery. Images/video support up to 10 MiB, AMR voice
up to 2 MiB, and ordinary files up to 20 MiB; oversize specialized media within
the file bound is sent as a file. Transfers have a three-minute deadline. Group
media also requires a live owner callback and native success acknowledgement;
WeCom can reject additional responses after its callback/stream window closes.
There is no fallback to broadcasting to a group or addressing another user.

The protocol follows the [official WeCom smart-bot SDK](https://github.com/WecomTeam/aibot-node-sdk)
and the pinned reference's smart-bot plugin. Tests exercise a real loopback
WebSocket with fake credentials, subscription rejection, duplicate/restart
handling, group owner routes, upload acknowledgements and encrypted media-only
messages. No live WeCom account was contacted. Interactive cards and reactions
remain unsupported.
