# Review fixture: missing none guard

This diff introduces a regression by removing the `None` handling path from
`src/profile.py`. A good review should flag the unsafe assumption and explain
the resulting failure mode.
