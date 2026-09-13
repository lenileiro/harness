# Native Google Chat, Feishu/Lark, and Teams attachments

These transports accept attachments only from their authenticated webhook events,
after the configured user/conversation policy admits the message. The webhook
persists descriptors before acknowledging; the queue downloads bytes before
calling the scoped gateway receiver. Text replies and approval commands keep
their existing routes. Transfers are tested offline with HTTP fixtures; no live
Workspace, Feishu/Lark, or Microsoft tenant was used.

Inbound transfers allow at most 16 attachments and 20 MiB in aggregate. Downloads
stream with a 60-second deadline, reject compressed transport encodings, and never
follow redirects. Outbound files require inline bytes and are limited to 20 MiB
each. Neither supplied display URLs nor local paths are generic fetch sources.
Credential-bearing transfer URLs are redacted from HTTP logging.

## Google Chat

Chat-uploaded files use `attachmentDataRef.resourceName` and the authenticated
Chat media endpoint. Attachment names must belong to the received message and
space; `thumbnailUri` and `downloadUri` are ignored. The existing service account
and `chat.bot` scope support these downloads. Drive-linked attachments are
explicitly unsupported: this adapter neither requests Drive permissions nor
attempts to fetch Google Drive display links. See Google's [download API](https://developers.google.com/workspace/chat/api/reference/rest/v1/media/download)
and [attachment resource](https://developers.google.com/workspace/chat/api/reference/rest/v1/spaces.messages.attachments).

Google requires **user authentication** for native uploads. Keep `token_env` as
the service-account JSON used for bot text messages, and optionally set
`app_token_env` to an environment variable containing this authorized-user JSON:

```json
{
  "type": "authorized_user",
  "client_id": "YOUR_OAUTH_CLIENT_ID",
  "client_secret": "YOUR_OAUTH_CLIENT_SECRET",
  "refresh_token": "YOUR_USER_REFRESH_TOKEN",
  "scopes": ["https://www.googleapis.com/auth/chat.messages.create"]
}
```

Obtain the refresh token through the operator's normal Google OAuth consent flow.
The adapter refreshes against Google's fixed token endpoint; it does not launch
an OAuth browser or grant additional scopes. `chat.messages` is also accepted.
The credential's scope declaration is checked locally, any returned scope is
checked on refresh, and Google remains authoritative about the actual grant.
Without these explicit user credentials, attachment uploads fail before any
upload request. **Attachment messages are posted as that authorized user**;
ordinary text remains posted as the bot.

Uploads use multipart/related metadata and bytes, then create a message with the
returned attachment reference. Upload space and reply thread match the queued
destination; `REPLY_MESSAGE_OR_FAIL` prevents fallback to another thread. The
message create request retains a stable request ID. Interrupted uploads can
leave an unused upload reference; uncertain sends require operator inspection.
See Google's [media upload API](https://developers.google.com/workspace/chat/api/reference/rest/v1/media/upload)
and [upload guide](https://developers.google.com/workspace/chat/upload-media-attachments).

## Feishu and Lark

The existing app secret, tenant access token, verification token, and Encrypt Key
continue to apply. Enable the relevant IM message/resource permissions in the
app console. The same implementation uses the appropriate Feishu or Lark host.

Inbound `image`, `file`, `audio`, and `media` events and images embedded in `post`
messages download through the message ID plus native image/file key. The adapter
does not fetch rich-text links or video thumbnails as substitute file contents.
Outbound images (up to 10 MiB) use the native image upload and image message APIs. Opus audio
uses the file upload API with the Opus file type and an audio message; an Ogg
container must identify Opus. Known duration is forwarded in milliseconds.
Other audio and video formats are delivered as ordinary downloadable files;
there is no implicit transcoding. Rich text contributes its text and embedded
images, without reproducing every formatting element.

Uploaded resources are sent to the exact chat or replied to the original root
with `reply_in_thread`; delivery UUIDs remain stable across retries. See the
official [message resource API](https://open.feishu.cn/document/server-docs/im-v1/message/get-2),
[image upload API](https://open.feishu.cn/document/server-docs/im-v1/image/create),
[file upload API](https://open.feishu.cn/document/server-docs/im-v1/file/create),
and [generated SDK request contract](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/api/im/v1/model/get_message_resource_request.py).

## Microsoft Teams

Native inline image downloads use only the authenticated Connector attachment
route or Microsoft's regional Skype media resource route. Bot credentials are
sent only to those native endpoints. Personal-chat file cards use the signed
OneDrive/SharePoint `downloadUrl` supplied in the authenticated activity, without
a Bot Connector bearer token. HTTPS, allowed Microsoft file hosts, standard TLS
ports, and no redirects are required. Links in chat text and arbitrary image
URLs are never downloaded.

Small outbound images (up to 20 KiB raw bytes) use a native inline data attachment.
Larger images and other files use the personal-chat file-consent workflow. Add
`"supportsFiles": true` to the Teams app manifest's bot entry and include the
`personal` scope. An authenticated conversation must already exist.

1. The outbox persists a random consent identifier, recipient, conversation,
   service URL, filename, file hash, and 15-minute expiry, then sends a native
   file consent card.
2. The delivery remains **pending** until that recipient accepts. Unrelated
   deliveries continue. An authenticated `fileConsent/invoke` must match the
   original owner, tenant, conversation, bot recipient, and service route; it
   never dispatches an agent run.
3. The outbox uploads bytes to the supplied OneDrive/SharePoint upload session
   without a bot token. Only a completed upload response advances to the native
   file info card. A successful upload is recorded before sending that card, so
   retrying the card does not repeat the upload.
4. Only after the info card succeeds does the delivery become **sent**. Declined
   or expired consent becomes **failed**, with a specific outbox error. Consent
   state survives restart. Ambiguous transfer failures retain the existing
   uncertain-delivery workflow; inspect the destination before retrying.

File consent works only in personal chats. Group/channel file attachments require
separate Microsoft Graph permissions and are explicitly unsupported by this Bot
Connector transport; small inline images still work. The adapter does not manage
OneDrive file deletion or Microsoft national-cloud deployments. See Microsoft's
[Teams file APIs](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/how-to/bots-filesv4),
[file consent configuration](https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/teams/teams-files),
and [upload-session byte transfer contract](https://learn.microsoft.com/en-us/graph/api/driveitem-createuploadsession?view=graph-rest-1.0).

## Verification

`packages/cli/tests/test_native_channel_media.py` exercises native request bodies,
per-platform tokens, media-only ingress, both Feishu/Lark hosts, exact reply
routes, Drive rejection, owner-bound Teams consent, restart and retry behavior,
and download bounds/redirect rejection. Existing webhook and enterprise channel
tests continue to cover real JWT/signature verification, allowlists, approval
commands, and durable acknowledgement. Production delivery still needs an
operator-run smoke test in each configured account.
