"""Offline integration server launched through the real SDK stdio transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("offline-harness-test", log_level="ERROR")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(text: str) -> dict[str, str]:
    return {"echo": text}


@server.tool()
def environment() -> dict[str, str | None]:
    return {
        "explicit": os.environ.get("EXPLICIT"),
        "literal": os.environ.get("LITERAL"),
        "unforwarded": os.environ.get("HARNESS_MCP_TEST_UNFORWARDED"),
        "cwd": str(Path.cwd()),
    }


@server.tool()
def fail() -> str:
    raise ValueError("deliberate server failure")


@server.tool()
async def slow(delay: float = 60) -> str:
    await asyncio.to_thread(_record_call)
    await asyncio.sleep(delay)
    return "finished"


def _record_call() -> None:
    with Path("started").open("a") as handle:
        handle.write("call\n")


@server.resource("test://message")
def message() -> str:
    return "offline resource"


@server.resource("test://greeting/{name}")
def greeting(name: str) -> str:
    return f"Hello {name}"


@server.prompt()
def summarize(subject: str) -> str:
    return f"Summarize {subject}"


if __name__ == "__main__":
    Path("server.pid").write_text(str(os.getpid()))
    server.run(transport="stdio")
