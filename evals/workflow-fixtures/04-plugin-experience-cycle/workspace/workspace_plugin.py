from typing import ClassVar

from harness.core.extensions import ToolProvider
from harness.core.tool_entry import ToolSpec


class DemoProvider(ToolProvider):
    def specs(self):
        return [ToolSpec(name="demo_tool", factory=lambda _ctx: _DemoTool())]


class _DemoTool:
    name = "demo_tool"
    description = "Demo tool from local plugin"
    parameters_schema: ClassVar[dict[str, object]] = {"type": "object", "properties": {}}
    approval = "auto"

    async def __call__(self, call):
        return "demo"
