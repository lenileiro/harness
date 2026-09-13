import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.core import Agent, FailoverPolicy, RunRequest, ToolCall, ToolRegistry
from harness.core.skills import Skill, SkillError, SkillLibrary, SkillReadTool, install_skill

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


def make_skill(
    root: Path, name: str = "release-check", body: str = "Always inspect the release diff."
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: |\n  Check a release.\n  Use before publication.\n---\n{body}\n"
    )
    return path


def test_frontmatter_and_precedence(tmp_path):
    first = make_skill(tmp_path / "workspace")
    make_skill(tmp_path / "user", body="lower priority")
    invalid = make_skill(tmp_path / "user", name="Invalid")
    library = SkillLibrary.load([tmp_path / "workspace", tmp_path / "user"])
    assert library.get("release-check").directory == first
    assert len(library.shadowed) == 1
    assert str(invalid) in library.errors
    catalog = library.render_context()
    assert "Check a release" in catalog
    assert "Always inspect" not in catalog


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "nested/../../secret"])
def test_supporting_files_cannot_escape(tmp_path, path):
    skill = Skill.load(make_skill(tmp_path))
    with pytest.raises(SkillError):
        skill.read(path)


async def test_symlink_and_wrong_arguments_rejected(tmp_path):
    directory = make_skill(tmp_path)
    (directory / "secret").symlink_to(tmp_path / "private")
    (tmp_path / "private").write_text("must not disclose")
    tool = SkillReadTool(SkillLibrary.load([tmp_path]))
    result = await tool(
        ToolCall(id="x", name=tool.name, arguments={"name": "release-check", "path": "secret"})
    )
    assert result.is_error and "must not disclose" not in result.content
    bad = await tool(ToolCall(id="x", name=tool.name, arguments={"name": 1}))
    assert bad.is_error


def test_install_keeps_resources_and_refuses_overwrite(tmp_path):
    source = make_skill(tmp_path / "source")
    (source / "references").mkdir()
    (source / "references/check.md").write_text("Check versions")
    target = install_skill(source, tmp_path / "installed")
    assert Skill.load(target).read("references/check.md") == "Check versions"
    with pytest.raises(SkillError, match="already exists"):
        install_skill(source, tmp_path / "installed")


async def test_activation_is_persisted_and_restored_on_resume(tmp_path):
    make_skill(tmp_path)
    library = SkillLibrary.load([tmp_path])
    store = MockStorage()
    adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(call_id="c", name="skill_read", arguments={"name": "release-check"}),
            text_turn("done"),
        ],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=ToolRegistry(),
        storage=store,
        failover=FailoverPolicy(chain=["mock"]),
        skill_library=library,
    )
    async for _ in agent.run(RunRequest(prompt="Load release-check", model="m", session_id="s")):
        pass
    session = await store.get("s")
    assert session is not None
    assert session.metadata["active_skills"] == ["release-check"]
    from harness.core import ModelRequestEvent

    adapter.scripts = [text_turn("resumed")]
    events = [e async for e in agent.resume("s", prompt="continue")]
    request = next(e for e in events if isinstance(e, ModelRequestEvent))
    assert any("Active skill release-check" in (m.content or "") for m in request.messages)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO regression is POSIX-only")
@pytest.mark.parametrize("filename", ["SKILL.md", "reference.txt"])
def test_skill_fifo_is_rejected_without_blocking_discovery_or_read(tmp_path, filename):
    directory = make_skill(tmp_path)
    target = directory / filename
    target.unlink(missing_ok=True)
    os.mkfifo(target)
    # A child process and deadline make the regression fail instead of hanging
    # the suite if a blocking open is accidentally reintroduced.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from harness.core.skills import Skill; "
            "skill = Skill.load(Path(sys.argv[1])); skill.read(sys.argv[2])",
            str(directory),
            filename,
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode != 0
    assert "skill files must be regular files" in result.stderr


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires leaf no-follow support")
def test_skill_read_rejects_symlink_swapped_after_path_validation(tmp_path, monkeypatch):
    directory = make_skill(tmp_path)
    skill = Skill.load(directory)
    target = directory / "reference.txt"
    target.write_text("ordinary reference")
    private = tmp_path / "private.txt"
    private.write_text("private-sentinel")
    original_open = os.open

    def swap_before_open(path, flags, *args, **kwargs):
        if Path(path) == target:
            target.unlink()
            target.symlink_to(private)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(SkillError) as caught:
        skill.read("reference.txt")
    assert "private-sentinel" not in str(caught.value)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO regression is POSIX-only")
def test_skill_install_rejects_special_file_before_copying(tmp_path):
    source = make_skill(tmp_path / "source")
    os.mkfifo(source / "reference.txt")
    with pytest.raises(SkillError, match="unsupported package entry"):
        install_skill(source, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
