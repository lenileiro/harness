# Preserve context across multiple gateway chats without transcript bleed

Use the built-in Harness CLI and edit the real gateway implementation in this
workspace so a single WhatsApp user can continue related work from multiple chat
threads without replaying unrelated old messages.

Use the actual code under `packages/core/src/harness/core` and
`packages/cli/src/harness/cli`.

Do all of the following:

1. Add a durable shared user-level gateway memory object.
2. Persist shared active work and recent thread references separately from
   per-thread message history.
3. Update gateway command handling so reminder, mission, research, or report
   actions created in one thread become available as shared context for that
   same user in another thread.
4. Update `gateway converse` so it resolves:
   - current thread context
   - shared active work for that user
   - summaries from other recent threads for that user
   before calling the model.
5. Keep thread-local transcript history separate so unrelated chats do not
   blindly merge together.
6. Add deterministic tests that prove a second thread can continue related work
   from the first thread using the shared user context.
7. Run the focused tests when done.

Important:

- Do not solve this by loading every prior thread transcript into the model.
- Do not hand-edit generated `.harness/gateway/*` state files as the primary
  implementation.
- The design should be durable across gateway restarts.
- The second thread should receive useful shared work context even if it has no
  prior local `thread_context` yet.
- The implementation should stay transport-neutral at the storage and router
  layer even if the motivating example is WhatsApp.

Run the tests when done.
