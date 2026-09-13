# Accounts and service bundles

Named OpenAI-compatible providers can use independent API keys or an explicitly
configured OAuth device flow. Credentials live under the selected identity's
`auth/accounts` directory with private permissions. The credential identity
includes the provider name, OAuth endpoints, client ID, scopes, and inference
endpoint; changing any of those does not reuse another binding's token.

```toml
[provider.company]
driver = "openai-compatible"
base_url = "https://models.example.org/v1"
model = "company-model"
input_media = ["image"]

[provider.company.oauth]
device_authorization_endpoint = "https://login.example.org/device"
token_endpoint = "https://login.example.org/token"
client_id = "your-registered-public-client"
scopes = "inference:invoke"
```

Use `harness accounts login company`, `accounts status company`, and
`accounts logout company`. `--no-browser` prints the verification link/code for
an SSH session. Login is explicit; inference only reads or refreshes saved
credentials. Device polling respects the server's interval and `slow_down`.
Refresh is serialized across processes and rotates the saved refresh token.
Terminal or ambiguous refresh failures quarantine it, avoiding repeated replay
of a revoked grant. Temporary server failures retain credentials. Status never
prints tokens. Logout removes the local record; it does not revoke the remote
grant.

For API-key providers, replace the OAuth table with
`api_key_env = "COMPANY_MODEL_KEY"`. An empty `api_key_env` is an explicit
unauthenticated-endpoint selection. Provider failover can set a separate
`model` in each `[provider.NAME]` table.

## Nous-compatible bundle

The bundle implements the protocols in the pinned Hermes
[Portal OAuth flow](https://github.com/NousResearch/hermes-agent/blob/939e45c91d751fadd94dcd1b873ac3cb44846213/hermes_cli/auth_device_flow.py),
[refresh flow](https://github.com/NousResearch/hermes-agent/blob/939e45c91d751fadd94dcd1b873ac3cb44846213/hermes_cli/auth_nous.py),
and vendor routes. It has not been exercised against a paid account.

```sh
harness portal configure --client-id YOUR_REGISTERED_CLIENT --model YOUR_MODEL
harness portal login
harness portal info
```

Configuration is separate from login and does not contact any service.
`--route web --route speech` selects individual routes; omitting the option
configures web, images, speech, transcription, and browser. Existing provider or
portal sections require `--replace-existing`. Unrelated settings are preserved.
A valid public OAuth client registration and the provider's service entitlements
are required; Harness does not register an application or purchase a plan.

The persisted `[portal]` table accepts explicit `web_url`, `image_url`,
`audio_url`, and `browser_url` overrides for compatible infrastructure. Each
route uses the selected provider account; direct per-tool credentials are not
used while that route is selected. Route failures are surfaced without switching
accounts or providers.

| Route | Wire workflow | Local behavior |
| --- | --- | --- |
| Web | Firecrawl `/v2/search` and `/v2/scrape` | Bounded external content; public-URL check for extraction. |
| Images | FAL queue submit/status/result/cancel | Same-origin queue URLs, unique submission keys, bounded allowlisted artifact download, native image attachment and saved file. |
| Speech/transcription | OpenAI audio endpoints | Fresh account token per request, bounded workspace files and native audio artifacts. |
| Browser | Browser Use create/PATCH-stop and CDP | Created only by the approved browser tool, isolated browser context, owned-session cleanup. |

Generated image downloads carry no account Authorization header and must use
HTTPS on `artifact_hosts` (default `fal.media` and its subdomains). CDP endpoints
must use HTTPS/WSS on `browser_cdp_hosts` (default `browser-use.com` and its
subdomains). Configure extra domains explicitly if your provider returns a
different host. Signed CDP URLs remain private configuration. The browser's
server-side lifetime bounds abandoned sessions when a create response is lost;
no automatic create retry is attempted.

These are local trusted-agent integrations. The API and remote gateway retain
their separately restricted tool exposure and do not inherit this account's
managed tools merely because the profile configured them.

Offline tests cover device polling, concurrent refresh/rotation, token quarantine,
resource/profile binding, vendor HTTP headers, queue cancellation, artifact
persistence, and browser create/stop ownership. External account compatibility,
entitlements, provider pricing, and real cloud resources remain live checks.

## Honcho user modeling

Honcho is a distinct optional service, configured separately from the Portal
bundle and local persistent memory:

```toml
[honcho]
enabled = true
api_key_env = "HONCHO_API_KEY"
base_url = "https://api.honcho.dev"
workspace_prefix = "harness"
```

The implementation uses the [Honcho v3 API](https://honcho.dev/docs/v3/documentation/reference/sdk),
verified against the official `honcho-ai` 2.4.0 route and request schemas.
`honcho_status` inspects local export state and provides message references.
`honcho_sync` requests approval to send selected user/assistant text to the
configured service; system instructions, tool outputs, and binary attachments
are excluded. `honcho_chat` asks that service a question about the identity's
exported conversations, also through the normal approval gate. Background
reasoning and the quality of Honcho's answers remain the external service's
responsibility; no live account was queried during implementation.

Workspace identifiers are derived from the profile home, local workspace/user
scope, service endpoint, and configured prefix. Remote channel users do not
inherit these tools. Separate profiles and workspaces get separate remote
workspaces. A local SQLite ledger prevents replaying a transcript write after a
lost response or process interruption. Inspect with `harness honcho exports`;
after checking the remote service, use `honcho reconcile WORKSPACE REFERENCE
--outcome exported` or `--outcome not-exported` to record the result. Reconciliation
does not itself contact the service.
