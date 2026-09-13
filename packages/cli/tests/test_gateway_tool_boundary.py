"""Remote path guards protect shared state even when a tool action is approved."""

from types import SimpleNamespace
from typing import cast

import pytest

from harness.cli.gateway_tool_boundary import install_gateway_tool_boundary
from harness.core import Agent, ToolCall, ToolRegistry
from harness.core.tools import Tool
from harness.tools.fs import EditFileTool, GlobTool, ListDirTool, ReadFileTool, WriteFileTool


@pytest.fixture
def workspace(tmp_path):
    private = tmp_path / ".harness" / "gateway" / "victim"
    private.mkdir(parents=True)
    (private / "secret.txt").write_text("PRIVATE_SENTINEL")
    (tmp_path / "public.txt").write_text("public project")
    (tmp_path / "alias").symlink_to(private, target_is_directory=True)
    (tmp_path / "secret-alias.txt").symlink_to(private / "secret.txt")
    return tmp_path


def registry(workspace, *, local_only=False):
    tools = ToolRegistry()
    for tool_type in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GlobTool):
        tools.register(tool_type(cwd=workspace))
    install_gateway_tool_boundary(
        cast(Agent, SimpleNamespace(tools=tools)), workspace, local_only=local_only
    )
    return tools


async def invoke(tools, name, **arguments):
    return await tools.get(name)(ToolCall(id="call", name=name, arguments=arguments))


@pytest.mark.parametrize(
    "path",
    [
        ".harness/gateway/victim/secret.txt",
        "alias/secret.txt",
        "secret-alias.txt",
        ".HARNESS/gateway/victim/secret.txt",
    ],
)
async def test_remote_read_write_edit_cannot_access_private_state(workspace, path):
    tools = registry(workspace)
    for name, arguments in (
        ("read_file", {}),
        ("write_file", {"content": "overwritten"}),
        ("edit_file", {"old": "PRIVATE_SENTINEL", "new": "overwritten"}),
    ):
        result = await invoke(tools, name, path=path, **arguments)
        assert result.is_error
        assert "PRIVATE_SENTINEL" not in result.content
    assert (workspace / ".harness/gateway/victim/secret.txt").read_text() == "PRIVATE_SENTINEL"


async def test_remote_listing_and_globbing_do_not_disclose_private_names(workspace):
    tools = registry(workspace)
    for name, arguments in (
        ("list_dir", {}),
        ("glob", {"pattern": "**/*"}),
        ("glob", {"pattern": "alias/*"}),
    ):
        result = await invoke(tools, name, **arguments)
        assert not result.is_error
        assert ".harness" not in result.content
        assert "victim" not in result.content
        assert "secret" not in result.content
        assert "alias" not in result.content
    result = await invoke(tools, "list_dir", path="alias")
    assert result.is_error
    result = await invoke(tools, "glob", pattern=".harness/**/*")
    assert result.is_error


async def test_public_files_work_and_local_only_retains_access(workspace):
    tools = registry(workspace)
    assert not (await invoke(tools, "read_file", path="public.txt")).is_error
    assert not (
        await invoke(tools, "write_file", path="new.txt", content="new project file")
    ).is_error
    assert not (await invoke(tools, "edit_file", path="new.txt", old="new", new="updated")).is_error
    assert (workspace / "new.txt").read_text() == "updated project file"
    for name, arguments in (("list_dir", {}), ("glob", {"pattern": "*.txt"})):
        assert "public.txt" in (await invoke(tools, name, **arguments)).content
    local = registry(workspace, local_only=True)
    result = await invoke(local, "read_file", path=".harness/gateway/victim/secret.txt")
    assert not result.is_error and result.content == "PRIVATE_SENTINEL"


def test_installer_is_idempotent_and_keeps_custom_tools(workspace):
    tools = registry(workspace)
    custom = SimpleNamespace(name="fake_action")
    tools.register(cast(Tool, custom))
    wrapped = tools.get("read_file")
    install_gateway_tool_boundary(cast(Agent, SimpleNamespace(tools=tools)), workspace)
    assert tools.get("read_file") is wrapped
    assert tools.get("fake_action") is custom
