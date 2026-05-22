# Fix hyphenated user ID lookup

`get_user()` returns `None` for hyphenated IDs like `'abc-def'`. Simple IDs
like `'alice'` and `'bob'` work fine.

The validation regex was recently audited and updated to explicitly allow
hyphens. The bug is likely somewhere downstream of validation.

Fix it so hyphenated IDs resolve correctly. Run the tests when done.
