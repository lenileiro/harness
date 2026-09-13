# Continue a conversation on another channel

Handoff copies an explicit snapshot of a completed conversation to one exact destination in the same Harness workspace and profile. It does not link two users permanently or merge their memories.

1. In the destination conversation, send `/whoami`. Copy the returned JSON object with `transport`, `user_id` and `thread_id`.
2. In the source conversation, send `/handoff` followed by that JSON. The source must have a completed run with no pending questions or unfinished approvals. Answer or skip any clarification and continue the original session before copying it.
3. In the destination, send `/continue CODE` with the returned code, then send your next message.

For example, a destination identity might be:

```json
{"transport":"slack","user_id":"T_WORKSPACE:U_USER","thread_id":"T_WORKSPACE:D_DM:"}
```

The code expires after ten minutes and is valid only for that destination user and thread. `/handoff revoke CODE` in the source revokes an unclaimed code. Source and destination may both be group threads only when their configured channel policies allow that user; choose the destination deliberately.

The snapshot includes conversation messages, completed tool evidence and attachments, bounded to 32 MiB. Source configuration, system/persona instructions, private memories, notes, phases, task ownership, approval overrides and approval ledgers are excluded. A compacted conversation summary is copied as earlier conversational context. The new session has the destination's memory scope and remote tool boundary. Its provider/model match the source; that provider still needs to be available in the selected profile, and the destination's next run applies the ordinary capability and approval checks.

The source transcript remains available. The destination gets a fresh session and does not overwrite an existing transcript. A crash during import reserves one fixed session ID so a repeated `/continue` can finish; replay after completion returns that existing ID without replacing subsequent messages. Claimed imports may be completed after the normal issue window to recover a crash. At most ten unclaimed handoffs may be outstanding per source. Raw codes are not stored in the handoff ledger; messaging platforms and the ordinary channel inbox may retain the command itself.

This is explicit conversation transfer. It does not infer that equal display names or similar addresses identify the same person, and does not grant access to another channel's approval requests.
