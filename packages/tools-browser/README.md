# Browser tools

```python
from pathlib import Path
from harness.tools.browser import BrowserConfig, BrowserToolset

async with BrowserToolset(BrowserConfig(), cwd=Path.cwd()) as browser:
    tools = browser.tools
```

Install the browser once with `python -m playwright install chromium`.
`BrowserConfig(executable_path="...")` selects an existing Chromium executable.
`BrowserConfig(backend="cdp", cdp_url="wss://...")` connects to an operator-selected
CDP endpoint. A connection failure never launches a local browser. Every toolset
creates a fresh context; existing CDP pages, cookies, and profiles are not reused.
Closing it removes its context and disconnects. Cloud session acquisition and
provider billing are managed by the operator supplying that endpoint.

`browser_snapshot` reads the current tab. `browser` supports `navigate`,
`snapshot`, `click`, `type`, `select`, `press`, `back`, `tabs`, `new_tab`,
`close_tab`, `screenshot`, `upload`, and `download`. Actions return JSON with
tab IDs, accessible page content, and refs for interactive elements. Use the
latest ref for that tab or an unambiguous selector; refreshed snapshots retire
old refs. Screenshots include a model-visible `MediaAttachment` and a saved PNG.
Actions that can interact with external services retain prompt approval.

Uploads read explicit workspace paths and reject `.harness` private state,
including symlink aliases and paths escaping the workspace. Click a download
link, then use its returned `download_id` to save it under a unique directory
inside `artifacts/browser` (configurable). File sizes and retained download IDs
are bounded; partial artifact files are removed when saving fails or is
cancelled. Context downloads are temporary until explicitly saved.

Navigation accepts HTTP(S) URLs, and browser contexts block non-HTTP(S) routed
requests and service workers. The browser uses its host's network access,
including local development servers; it is not an egress-isolation boundary.
Model-facing page data remains untrusted. Dialogs are dismissed automatically.
The toolset exposes no arbitrary page JavaScript or personal browser profile
attachment. Desktop control is a separate capability.

Tests exercise real headless Chromium against a loopback HTTP fixture: readable
snapshots, stale refs, form interaction, tabs, upload/download, screenshot
attachments, and cancellation cleanup. CDP connection and isolated-context
lifecycle are tested with a fake provider; no cloud CDP service is contacted.
