# WhatsApp Cloud, BlueBubbles, and Microsoft Graph notifications

These adapters feed the existing scoped gateway and durable channel inbox/outbox.
Configure the selected profile, supply secrets through environment variables, then
run `harness channel run NAME --cwd /path/to/workspace`. Inspect deliveries
with `harness channel status NAME --cwd /path/to/workspace`. `retry` requires
an explicit message/delivery ID; inspect the destination first after an uncertain
send. A send acknowledgment means API acceptance, not delivery to a person's device.

Callbacks are acknowledged after local persistence. Account identities bind the
channel database; inbound user and conversation allowlists also apply at dispatch
and delivery. Reverse proxies must preserve raw signed bodies and query parameters,
enforce HTTPS externally, and disable credential-bearing query/access logs.

## WhatsApp Cloud

```toml
[channels.whatsapp_cloud]
token_env = "WA_ACCESS_TOKEN"
app_token_env = "WA_VERIFY_TOKEN"
signing_secret_env = "META_APP_SECRET"
phone_number_id = "123456789"
waba_id = "987654321"
api_version = "v23.0"
allowed_users = ["15551234567"]
allowed_channels = ["15551234567"]
webhook_path = "/whatsapp-events"
listen_host = "127.0.0.1"
listen_port = 8770
```

Use the WhatsApp sender ID digits, without a plus sign, for allowlists. `waba_id`
identifies the business account; `phone_number_id` is the Graph resource ID, not
the displayed telephone number. Set up the business assets, register the phone,
grant the token the required WhatsApp permissions, and subscribe the app to the
WABA and its `messages` webhook field in Meta. Set the public callback URL to the
configured path and the verification token to the value of `WA_VERIFY_TOKEN`.
Harness validates the GET challenge, requires the app-secret HMAC-SHA256 signature
on each POST, and rejects events for another WABA or phone. The Graph API version
is explicit and can be updated by the operator. See [Meta's official Cloud API
collection](https://www.postman.com/meta/whatsapp-business-platform/documentation/wlk6lh4/whatsapp-cloud-api)
and [webhook setup](https://developers.facebook.com/docs/graph-api/webhooks/getting-started/).

Text, interactive button/list replies, and image/audio/document/video/sticker
inputs are supported. Media downloads resolve numeric Graph IDs, allow only the
platform's authenticated download hosts, reject redirects, and cap aggregate input
at 20 MiB. Outbound inline uploads support image/audio/video/document messages;
images are capped at 5 MiB, audio/video at 16 MiB, and documents at 20 MiB.
Captions are preserved on inbound media. Outbound remote URLs, template campaigns,
phone onboarding, and status-driven delivered/read tracking are not implemented.
Free-form replies remain subject to Meta's messaging eligibility rules; an expired
reply window does not silently become a template send. Outbox IDs are passed as
opaque callback data for correlation, not claimed as server-side idempotency keys.

## BlueBubbles

```toml
[channels.bluebubbles]
token_env = "BLUEBUBBLES_PASSWORD"
app_token_env = "BLUEBUBBLES_WEBHOOK_TOKEN"
homeserver = "http://127.0.0.1:1234"
webhook_url = "http://127.0.0.1:8771/bluebubbles-events"
webhook_path = "/bluebubbles-events"
listen_host = "127.0.0.1"
listen_port = 8771
allowed_users = ["+15551234567"]
allowed_channels = ["iMessage;-;+15551234567"]
```

Run a BlueBubbles server on a Mac signed into iMessage. It must report
`computer_id` and `detected_imessage` through `/api/v1/server/info`; Harness binds
its durable state to that computer/account. The REST API uses its password query
parameter. Use a separate random webhook token. HTTP is allowed only for loopback
destinations; remote server and callback URLs require HTTPS. These requirements
use the [BlueBubbles REST/webhook interface](https://docs.bluebubbles.app/server/developer-guides/rest-api-and-webhooks)
and its [server API source](https://github.com/BlueBubblesApp/bluebubbles-server/tree/master/packages/server/src/server/api).

When `webhook_url` is configured, Harness registers the exact URL with
`?token=<URL-encoded webhook token>` for `new-message`. It reuses matching existing
registrations, tracks its owned registration across crashes, and deletes only its
own registration on shutdown. Cancellation waits for a late creation response so
the newly created registration can be cleaned up. If the server created a webhook
but its response was lost, inspect the server's registrations before retrying.
When `webhook_url` is omitted, register that same callback manually in BlueBubbles.
The URL must be reachable from the Mac; loopback refers to the Mac itself.

Text and bounded inline media uploads/downloads are implemented. The sender comes
from the authenticated event's handle; the exact chat GUID is the conversation.
Messages from the local account, receipt updates and tapback reactions do not
trigger a run. Group chat GUIDs contain `;+;`. To enable groups, set `allow_groups`
and either configure `username` for an explicit `@alias` mention or set
`require_mention = false`. No default wake word is inferred. Bare addresses are not
resolved into groups, and new chats are not created implicitly. Private-API typing,
read receipts, editing and reactions are not implemented. A send's `tempGuid`
correlates with its outbox ID; ambiguous failures still require operator review.

## Microsoft Graph basic notifications

```toml
[channels.msgraph_webhook]
token_env = "GRAPH_CLIENT_STATE"
tenant_id = "TENANT_ID"
subscription_id = "SUBSCRIPTION_ID"
accepted_resources = ["users/USER_ID/messages"]
allowed_users = ["TENANT_ID:SUBSCRIPTION_ID"]
webhook_path = "/graph-events"
listen_host = "127.0.0.1"
listen_port = 8772
```

`GRAPH_CLIENT_STATE` is a dedicated random 32–128 byte secret, not an OAuth access
token. Create or renew a Graph subscription separately, using the same clientState
and public callback URL with `includeResourceData = false`. Harness answers Graph's
POST validation challenge as plain text. For first-time registration, start with
an explicit placeholder subscription ID; validation works, but notifications from
real subscriptions are rejected. After Graph returns the subscription ID, update
`subscription_id` and `allowed_users`, then restart. The durable account identity is
the tenant; every notification, sender identity and conversation also binds its
subscription. Changing tenants requires a separate workspace/profile.

Configure resource prefixes to match the notification's actual `resource` path.
Matching is case-sensitive and segment-bound; `users/USER_ID/messages-extra` does
not match `users/USER_ID/messages`. For OData-style resource paths, configure the
actual prefix as delivered. Resource data is metadata only: no arbitrary URL fetch
or OAuth impersonation occurs. Rich encrypted notifications are rejected. See
[Microsoft's webhook protocol](https://learn.microsoft.com/en-us/graph/change-notifications-delivery-webhooks)
and [rich notification authentication requirements](https://learn.microsoft.com/en-us/graph/change-notifications-with-resource-data).

The entire notification batch must pass clientState, tenant, subscription and
resource checks before work is queued. ClientState and unrelated payload fields
never enter transcripts. Stable created/deleted notifications and versioned updates
are deduplicated durably. Unversioned updates are preserved individually because
identical metadata can describe distinct edits; an upstream retry can therefore
repeat analysis. Lifecycle notices are persisted as `subscription_lifecycle` and
require operator subscription renewal or investigation; Harness does not own or
automatically renew external subscriptions.

Graph notifications have **no remote reply destination**. Gateway responses are
saved locally and appear in local channel status output under `local_responses`.
Here an outbox `sent` status means durable local delivery, never an email, Teams
post, or Graph API send. Remote/private status surfaces omit response text by
default. Media replies have no Graph delivery route and fail explicitly.

## Verification limits and reference

Automated tests use fake HTTP APIs and actual loopback webhook listeners. They
cover raw-body signatures, account/subscription boundaries, duplicate callbacks,
restart persistence, outbox uncertainty, bounded platform media, and cancellation
and ownership of BlueBubbles registration. No real Meta, Apple, BlueBubbles or
Microsoft account was connected and no external message was sent.

The compared reference is Hermes commit
[`939e45c91d751fadd94dcd1b873ac3cb44846213`](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213/gateway/platforms),
including its `whatsapp_cloud.py`, `bluebubbles.py`, and notification-only
`msgraph_webhook.py`. Platform APIs, account eligibility, and server versions still
need verification against the operator's selected live deployment.
