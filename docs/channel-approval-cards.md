# Native channel approval controls

Telegram, Discord, Slack, Feishu/Lark and Teams render native **Approve** and
**Deny** buttons for pending tool actions. The full request, arguments
and expiry are sent first; a small decision card follows in the same
conversation. Approve authorizes that one stored action. Session-wide or permanent
grants are not offered through these controls.

The buttons use the same SQLite ApprovalStore and scoped gateway resolution as
`approve APPROVAL_ID` and `deny APPROVAL_ID`. They do not invoke tools directly or
accept a command, tool arguments, session ID or permission scope from the button
payload. Only actual pending records owned by a persisted gateway binding can
produce cards; model-generated text or an invented approval ID cannot.

Each card persists two random opaque tokens and binds them to the approval,
runtime session, original user, channel/thread and native sent-message ID. An
authenticated callback must match those values and the current allowlist. Its
record and gateway binding are checked again before acceptance. Consuming either
choice invalidates both choices and any copies of that action's card while
atomically inserting one ordinary decision command in the channel inbox. Callback
acknowledgement follows persistence, before model continuation. Restart, duplicate
callbacks, another user in the same group, altered native message IDs, stale
sessions and expired/resolved requests cannot create another accepted decision. Gateway
approvals expire 15 minutes after their original request; rendering another card
does not extend that lifetime.

Cards also accompany lifecycle approval notifications, including scheduled prompt
jobs, when the destination has a persisted inbound conversation. A target with no
prior inbound context receives the ordinary text request until its owner chats
with the bot. `approvals` lists current requests and can render pending cards.
Only one active card is queued per action; retrying delivery does not create
additional approval authority.

A queued decision is not a claim that execution completed. The normal gateway
lock, atomic approval resolution and replay claim still govern continuation. If
a conversation is busy or continuation becomes uncertain, inspect the reported
state and use the existing text/session controls. A send interrupted before a
native message receipt leaves the card unbound and the outbox uncertain; inspect
the destination before explicit retry. Old buttons can remain visible, but their
consumed or expired tokens cannot authorize work.

## Platform setup

- **Telegram:** the existing long poll includes `callback_query` updates. Inline
  keyboards stay on the original chat/topic. The callback sender and bot-authored
  message must match; inline-message callbacks without that context are rejected.
  See the [Bot API callback contract](https://core.telegram.org/bots/api#callbackquery).
- **Discord:** action-row buttons arrive through authenticated Gateway
  `INTERACTION_CREATE`. Keep the application's external Interactions Endpoint URL
  unset when using this Gateway path. The callback application, bot message,
  actor and channel are checked, and its short-lived interaction token is used
  only at Discord's fixed callback endpoint. Replies are ephemeral acknowledgements.
  See [interaction delivery and responses](https://docs.discord.com/developers/interactions/receiving-and-responding).
- **Slack:** enable Interactivity for the Socket Mode app. Block Kit button
  actions arrive on its authenticated socket and retain workspace/channel/root
  timestamp ownership. The envelope is acknowledged after durable insertion.
  See [Socket Mode interactive features](https://docs.slack.dev/apis/events-api/using-socket-mode/#using-interactive-features).
- **Feishu/Lark:** configure the `card.action.trigger` callback on the same signed
  callback listener, using the configured verification token and Encrypt Key.
  The operator's tenant/open ID, original chat and native card message ID are
  checked; the callback returns a success/error toast. See the official
  [card callback SDK schema](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/event/callback/model/p2_card_action_trigger.py).
- **Teams:** the existing Bot Connector route accepts Adaptive Card 1.4
  `Action.Execute` callbacks under `adaptiveCard/action`; legacy `Action.Submit`
  data is also understood. The Connector JWT, tenant, signed service URL, bot
  recipient, actor, conversation and original card ID must match. Unknown verbs
  are rejected. See [Teams card actions](https://learn.microsoft.com/en-us/microsoftteams/platform/task-modules-and-cards/cards/cards-actions).

This implements the pinned reference's native decision-button workflow with
Harness's existing single-action grant semantics. Clarification questions,
platform command menus, arbitrary interactive forms and reactions are separate
features. Tests cover all six transport names, signed Feishu/Teams callbacks,
a real local Slack socket, both decisions, concurrent choices, restart,
notification deduplication and a single gateway continuation. No live account,
model call or external message was used for verification.
