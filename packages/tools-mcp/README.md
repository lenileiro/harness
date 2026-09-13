# Harness MCP client

This package uses the official MCP Python SDK (`mcp==1.30.0`) for stdio and
Streamable HTTP connections. Configure servers in the Harness TOML file:

```toml
[mcp.servers.local]
transport = "stdio"
command = "/absolute/path/to/python"
args = ["/absolute/path/to/server.py"]
env_from = { SERVICE_TOKEN = "MY_SERVICE_TOKEN" }
include_tools = ["read*", "search"]
exclude_tools = ["read_private"]
approval = "prompt"
timeout = 30
expose_resources = true
expose_prompts = true

[mcp.servers.remote]
transport = "streamable-http"
url = "https://example.com/mcp"
bearer_token_env = "MY_MCP_TOKEN"
headers_from = { X-Organization = "MY_ORGANIZATION" }
```

`env` and `headers` contain literal values. `env_from`, `headers_from` and
`bearer_token_env` explicitly name host environment variables; missing references
fail before connection. Stdio inherits the SDK's minimal operating-system
environment plus the configured variables, rather than the complete host
environment. Relative server working directories resolve against the run's cwd.
Server programs run locally without an added OS sandbox. HTTP clients do not
inherit proxy environment settings.

For OAuth, set `oauth = true` on a Streamable HTTP server instead of a bearer
credential. Optional `oauth_scopes` and `oauth_redirect_port` bind its grant.
Run `harness mcp login SERVER` for the explicit browser/loopback PKCE flow;
`mcp logout SERVER` deletes local tokens. Tokens, expiry, verified issuer and
refresh endpoint persist privately inside the selected Harness profile. Refresh
rotation is serialized across processes. Old credentials without a bound issuer
or previously verified token endpoint require reauthorization.
The SDK is pinned because the persistence wrapper uses its OAuth lifecycle hooks;
upgrading it requires rerunning the discovery, PKCE, refresh and cancellation fixtures.

`harness mcp list` inspects configuration without connecting. `harness mcp check`
connects enabled servers, lists discovered tools and closes connections. Both
accept `--config` and `--json`; check also accepts a server name and `--cwd`.

Tools appear as `mcp__SERVER__tool__NAME`. Invalid characters and long names use
stable hashed aliases within the provider's 64-character name limit. Include and
exclude filters match original tool names with shell-style patterns. Server
annotations never grant approval; the configured default and Harness's ordinary
approval policy apply. Optional resources and prompts are exposed through
`list_resources`, `read_resource`, `list_prompts` and `get_prompt` tools only when
the server advertises the corresponding capability.

Text and structured results reach the model; structured results also appear in
tool metadata. Images/audio/blob resources become bounded native attachments,
with provider capability checks, instead of base64 strings in ordinary text.
Embedded text resources remain readable text. Output text is bounded and marks
truncation. Server-reported tool errors remain errors.

```python
from harness.core.tools import ToolRegistry
from harness.tools.mcp import MCPServerConfig, MCPToolset

registry = ToolRegistry()
servers = [MCPServerConfig(name="local", command="python", args=("server.py",))]
async with MCPToolset(servers, cwd=".", registry=registry) as toolset:
    # Run an Agent with this registry inside the context.
    print([tool.name for tool in toolset.tools])
```

Connections and registrations belong to the context. Errors during initialization
clean up previously opened servers. Calls are serialized per server. A transport
failure, timeout or cancellation closes that server connection and never replays
the tool request: a remote side effect may already have happened. Cancellation
closes the SDK session and local subprocess; it cannot establish whether remote
work was undone. A new run can reconnect explicitly.

See the [official SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.30.0)
and its [Streamable HTTP client](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/streamable_http.py).
