import pytest

from harness.cli.common import _build_adapter
from harness.cli.config import HarnessConfig
from harness.core import ConfigurationError


async def test_custom_provider_uses_own_name_key_and_capabilities(monkeypatch):
    monkeypatch.setenv("PRIVATE_ENDPOINT_KEY", "endpoint-token")
    monkeypatch.setenv("OPENAI_API_KEY", "other-identity")
    config = HarnessConfig(
        provider_settings={
            "company": {
                "driver": "openai-compatible",
                "base_url": "https://models.example.test/v1",
                "api_key_env": "PRIVATE_ENDPOINT_KEY",
                "input_media": ["image"],
                "max_context_tokens": 32000,
            }
        }
    )
    adapter = _build_adapter("company", base_url=None, config=config)
    assert adapter.name == "company"
    caps = await adapter.capabilities()
    assert caps.input_media == ["image"] and caps.max_context_tokens == 32000
    monkeypatch.delenv("PRIVATE_ENDPOINT_KEY")
    with pytest.raises(ConfigurationError, match="PRIVATE_ENDPOINT_KEY"):
        _build_adapter("company", base_url=None, config=config)
