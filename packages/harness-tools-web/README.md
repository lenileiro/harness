# harness-tools-web

HTTP fetch tool for Harness agents.

Tools:
- `fetch_url(url, timeout=15)` — GET a URL, return body (with size cap) and headers

Default approval policy: **prompt**. MIME types and size caps are configurable.
