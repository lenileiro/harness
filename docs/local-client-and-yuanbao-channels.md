# SimpleX, Buzz, Raft, Photon and Yuanbao

These adapters share Harness's scoped conversations, approval inbox, durable
message deduplication and outbox. Run them with `harness --profile NAME channel
run CHANNEL --cwd WORKSPACE`; `channel run-all` starts the configured transports
under that one profile. The commands below are operator setup instructions. No
external accounts, model calls or messages were used during implementation tests.

## SimpleX

Run a dedicated, already configured SimpleX Chat daemon with its local WebSocket
API enabled. Keep its active user fixed for the lifetime of the Harness runner:

```toml
[channels.simplex]
homeserver = "ws://127.0.0.1:5225"
username = "1" # numeric SimpleX active user ID
allowed_users = ["contact:23"]
allowed_channels = ["contact:23"]
```

Harness verifies `/user`, uses correlation IDs, and sends through the daemon's
numeric contact/group targets. Text is JSON-encoded into `/_send`, so message
content cannot become another daemon command. The daemon must listen on loopback.
For groups, explicitly allow `group:GROUP_ID:member:MEMBER_ID` users and
`group:GROUP_ID` channels, set `allow_groups=true`, and configure mention policy.
Group membership identities remain separate from direct-contact identities.
`require_mention=true` requires the daemon's `userMention` metadata; if that is not
provided by your daemon, use an explicitly allowed group with
`require_mention=false`.

Harness preserves events received during identity checks and deduplicates replay
after restart. It does not fetch old history or automatically accept contact/group
invitations. Delivery requires the daemon's correlated sent-item acknowledgment;
an interrupted send requires operator inspection. Native files, reactions and
invitation onboarding remain unsupported. Protocol reference:
[SimpleX Chat API](https://github.com/simplex-chat/simplex-chat/blob/stable/docs/api/README.md).

## Buzz

Install and authenticate the official Buzz CLI for the intended relay identity.
Harness supplies the explicitly named credentials to a bounded CLI child:

```toml
[channels.buzz]
command = "buzz"
homeserver = "https://YOUR_BUZZ_RELAY"
token_env = "BUZZ_PRIVATE_KEY"
app_token_env = "BUZZ_AUTH_TAG" # only if required by your relay
allowed_users = ["OWNER_PUBLIC_KEY_HEX"]
allowed_channels = ["CHANNEL_ID"]
allow_groups = true
require_mention = true
```

The CLI verifies the configured identity and channel access. Harness checks exact
public-key mention tags, retains root-thread IDs and waits for an accepted native
event receipt. The first history read establishes a baseline. Later reads use a
persisted inclusive timestamp plus message-ID deduplication. A full 200-event
recovery page stops progress with the cursor preserved; inspect the gap before
resuming. Private-key signing and relay event validation are delegated to the
official CLI; Harness does not claim independent signature verification of its
normalized JSON output. Native files, reactions, channel discovery and automatic
DM discovery remain unsupported. Protocol reference: [Block Buzz](https://github.com/block/buzz).

## Raft

Raft's pinned Hermes integration is a content-free wake bridge. Harness keeps
that contract and adds bounded, explicitly scoped CLI tools for handling work:

```toml
[channels.raft]
command = "raft"
username = "work" # existing Raft CLI profile
app_id = "RAFT_AGENT_ID"
homeserver = "https://YOUR_RAFT_SERVER"
token_env = "RAFT_BRIDGE_TOKEN" # separate random value, at least 32 characters
listen_host = "127.0.0.1"
listen_port = 8783
webhook_path = "/raft-wake"
allowed_users = ["raft:work"]
allowed_channels = ["work"]
allowed_targets = ["#project", "dm:@operator"]
```

Harness checks `raft --profile work auth whoami` against the configured server,
agent and profile. It owns the `agent bridge` subprocess and accepts only
authenticated, bounded wake metadata. Sender text and credentials are not accepted
as wake fields. Native user identity is not inferred from metadata-only events.

Only a configured Raft conversation receives `raft_manual`, `raft_check`,
`raft_read` and `raft_send`. Tools invoke fixed argv commands under the same
profile, with deadlines and output limits. Inbox draining, history reads that
record CLI context, and outgoing messages require ordinary reviewed approvals.
`allowed_targets` constrains read/send targets and their native threads. Shell
execution and arbitrary Raft commands are not exposed.

Wake replies and approval notices are saved **locally**, not sent as Raft messages.
Inspect them using `harness channel status raft --cwd WORKSPACE`. The local
operator can approve and resume that exact binding with:

```sh
harness gateway receive --cwd WORKSPACE --transport raft --user raft:work \
  --thread work --message "approvals"
harness gateway receive --cwd WORKSPACE --transport raft --user raft:work \
  --thread work --message "approve APPROVAL_ID"
```

Actual Raft messages are sent only by an approved `raft_send` call. Activity
mirroring is disabled; its drain endpoint reports an empty stream. Native media
and the reference's broader activity mirroring remain separate work. Reference:
[pinned Raft plugin](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213/plugins/platforms/raft).

## Photon iMessage

Photon uses its official Spectrum SDK in an owned Node process. Install Node 20+
and the helper's pinned dependency once (dependency install scripts are disabled):

```sh
PHOTON_BRIDGE_DIR=$(python -c 'from harness.cli.channels.photon import BRIDGE; print(BRIDGE.parent)')
npm install --prefix "$PHOTON_BRIDGE_DIR" --ignore-scripts --omit=dev --no-audit --no-fund
```

The helper manifest pins `spectrum-ts` to 12.8.0. Node's runtime receives only the
selected Photon project credentials and basic process environment. SDK logs are
suppressed, and stdout carries bounded structured messages. Harness waits for
the SDK's send receipt and terminates the child when the channel stops.

Create or select a project and register your phone through Photon device login:

```sh
harness --profile work channel photon-setup --cwd WORKSPACE \
  --phone +15551234567 --create-project "Harness"
# Or reuse an explicit project:
harness --profile work channel photon-setup --cwd WORKSPACE \
  --phone +15551234567 --project PROJECT_ID
```

Setup prints a device authorization URL/code and a usable configuration snippet.
It validates dashboard and project access, reuses an unambiguous existing project,
reads the current project secret without rotating it, and registers the phone
only if absent. It sends no invite and does not purchase/provision a paid line.
Credentials are written to an explicit workspace-local account file with private
permissions, no symlink traversal and no overwrite unless `--replace` is given.
The dashboard token is not persisted. An assigned iMessage number is printed when
available; otherwise finish line setup in the Photon dashboard.

```toml
[channels.photon]
app_id = "PROJECT_ID"
account_file = "/WORKSPACE/.harness/channels/photon-account.json"
allowed_users = ["SPECTRUM_USER_ID"]
```

Alternatively set `token_env="PHOTON_PROJECT_SECRET"` for externally managed
credentials. Direct messages retain the SDK space ID and sender user ID. Group
spaces require explicit group permissions and `require_mention=false` because
this bridge does not yet expose native mention metadata. Text-only support is
implemented; streaming edits, media, reactions, rich links and native presence
remain incomplete compared with the reference Photon plugin. Protocol references:
[Spectrum SDK](https://github.com/photon-hq/spectrum-ts),
[Photon CLI](https://github.com/photon-hq/cli).

## Yuanbao

Create a bot in Tencent Yuanbao and configure its issued app key/secret:

```toml
[channels.yuanbao]
app_id = "YUANBAO_APP_KEY"
token_env = "YUANBAO_APP_SECRET"
allowed_users = ["OWNER_ACCOUNT_ID"]
allowed_channels = ["dm:OWNER_ACCOUNT_ID"]
```

Harness signs the HTTPS token exchange, verifies the returned bot identity, then
performs binary WebSocket authentication against Tencent's fixed endpoint. It
uses the published compatible terminal type 16 with Harness version metadata;
it does not claim the Hermes terminal identity. Authentication rejects an existing
same-device connection instead of taking it over. Reconnects renew credentials
and reject a changed bot identity.

Frames use Tencent's published protobuf fields, bounded to 1 MiB. Authenticated
pushes are persisted before acknowledgment, and repeated message IDs remain
deduplicated after restart. Heartbeats correlate replies and adapt to the server's
interval. Outgoing requests retain a deterministic delivery message ID and require
both access-layer and business-layer success. Unknown send outcomes stay uncertain
for operator inspection; they are not automatically repeated.

Group reception requires `allow_groups=true`, a `group:GROUP_CODE` allowlist entry
and an exact native bot mention when `require_mention=true`. A private message
originating from a group retains its private context. This implementation supports
text and native mention elements; media, forwarded records, quote context,
streaming edits, reactions and device-conflict takeover are not implemented.
Protocol reference: [Tencent's Yuanbao plugin and protobuf schemas](https://github.com/Tencent/yuanbao-openclaw-plugin).
