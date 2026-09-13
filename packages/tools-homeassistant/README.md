# Home Assistant tools

Harness uses the [official Home Assistant REST API](https://developers.home-assistant.io/docs/api/rest/)
with one explicitly named bearer-token environment variable. It reads entity
states and service descriptions, and invokes granted services through the Harness
tool approval path. This package does not write synthetic states or manage HA
configuration.

```toml
[homeassistant]
enabled = true
base_url = "http://homeassistant.local:8123"
token_env = "HOME_ASSISTANT_TOKEN"
entities = ["light.desk", "sensor.office"]
services = ["light.turn_on"]

[homeassistant.service_fields]
"light.turn_on" = ["brightness"]
```

Supply the token through the selected environment reference; keep its value out
of configuration. The fixed endpoint may be on a private LAN. Redirects and
environment proxies are disabled. Connection setup performs no network request.

The tools are `homeassistant_list_entities`, `homeassistant_get_state`,
`homeassistant_list_services`, and, when entity/service grants are nonempty,
`homeassistant_call_service`. Service calls default to approval and accept only
exact configured entity IDs, services, and additional data fields. Wildcards,
`all`, and alternate area/device/target selectors are unavailable. Read tools
default to automatic execution. Returned changed-state lists exclude ungranted
entities. Operators must still account for the downstream behavior of any
granted script, automation, group, or service; these grants do not sandbox HA.

Requests have response-size and time limits. A failed or interrupted mutation
may have reached HA, so inspect state before deciding to submit another action.
The client never automatically retries a service call.

Library use:

```python
from harness.tools.homeassistant import HomeAssistantConfig, HomeAssistantToolset

async with HomeAssistantToolset(HomeAssistantConfig.model_validate(settings)) as managed:
    for tool in managed.tools:
        registry.register(tool)
    # Execute through Agent while the managed scope remains open.
```

Offline tests use HTTPX MockTransport and a real Harness Agent approval inbox;
they send no requests to real devices.
