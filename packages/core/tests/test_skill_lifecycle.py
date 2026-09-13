import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from harness.core import (
    Agent,
    FailoverPolicy,
    InboxApprovalHandler,
    RunRequest,
    ToolCall,
    ToolRegistry,
)
from harness.core.activity import ActivityEvent
from harness.core.schemas import Session, SessionStatus
from harness.core.skill_lifecycle import (
    SkillLifecycle,
    git_archive_url,
    record_skill_run,
    skill_evolution_tools,
)
from harness.core.skills import SkillError, SkillLibrary
from harness.storage.memory import InMemoryStorage

from .conftest import MockAdapter, MockTool, text_turn, tool_call_turn


def package(root: Path, body: str = "Check the output.") -> Path:
    path = root / "check-output"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(document(body))
    (path / "references").mkdir()
    (path / "references/example.txt").write_text("reference data")
    return path


def document(body: str) -> str:
    return f"---\nname: check-output\ndescription: Verify generated output.\n---\n{body}\n"


def archive(entries: list[tuple[str, bytes, bytes | None]]) -> bytes:
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as tar:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            if kind is not None:
                member.type = kind
                member.linkname = "../../private"
            tar.addfile(member, io.BytesIO(content))
    return result.getvalue()


def test_pinned_install_preserves_resources_and_records_content_provenance(tmp_path):
    lifecycle = SkillLifecycle(tmp_path / "skills")
    data = archive(
        [
            ("repo/skills/check-output/SKILL.md", document("Read the reference.").encode(), None),
            ("repo/skills/check-output/references/test.txt", b"reference", None),
            ("repo/skills/check-output/scripts/untrusted.sh", b"touch /must-not-execute", None),
        ]
    )
    requested = []

    def fetch(url):
        requested.append(url)
        return data

    state = lifecycle.install_git(
        "check-output",
        "https://github.com/org/repo.git",
        "a" * 40,
        "skills/check-output",
        fetch=fetch,
    )
    assert requested == [f"https://api.github.com/repos/org/repo/tarball/{'a' * 40}"]
    assert (tmp_path / "skills/check-output/references/test.txt").read_text() == "reference"
    assert state["revisions"][0]["provenance"] == {
        "kind": "git",
        "repository": "https://github.com/org/repo",
        "commit": "a" * 40,
        "path": "skills/check-output",
        "archive_sha256": hashlib.sha256(data).hexdigest(),
    }
    assert not state["modified"]
    with pytest.raises(SkillError, match="changed"):
        lifecycle.install_git(
            "check-output",
            "https://github.com/org/repo",
            "b" * 40,
            "skills/check-output",
            fetch=fetch,
        )


@pytest.mark.parametrize(
    "repository,commit",
    [
        ("https://github.com/o/r", "main"),
        ("https://github.com/o/r", "a" * 7),
        ("https://token@github.com/o/r", "a" * 40),
        ("http://github.com/o/r", "a" * 40),
        ("https://localhost/o/r", "a" * 40),
        ("https://github.com/o/r?token=secret", "a" * 40),
        ("https://github.com/o/../r", "a" * 40),
    ],
)
def test_git_sources_require_immutable_public_identity(repository, commit):
    with pytest.raises(SkillError):
        git_archive_url(repository, commit)


@pytest.mark.parametrize(
    "bad_name,kind",
    [
        ("repo/../private", None),
        ("/absolute", None),
        ("repo/link", tarfile.SYMTYPE),
        ("repo/hard", tarfile.LNKTYPE),
        ("repo/device", tarfile.CHRTYPE),
    ],
)
def test_unsafe_archive_fails_without_installing(tmp_path, bad_name, kind):
    data = archive([("repo/SKILL.md", document("safe").encode(), None), (bad_name, b"", kind)])
    lifecycle = SkillLifecycle(tmp_path / "skills")
    with pytest.raises(SkillError):
        lifecycle.install_git(
            "check-output", "https://github.com/o/r", "a" * 40, ".", fetch=lambda _: data
        )
    assert not (tmp_path / "skills/check-output").exists()


def test_archive_size_limits_include_compression_bombs(tmp_path, monkeypatch):
    import harness.core.skill_lifecycle as module

    data = archive(
        [("repo/SKILL.md", document("safe").encode(), None), ("repo/big", b"0" * 20000, None)]
    )
    lifecycle = SkillLifecycle(tmp_path / "skills")
    monkeypatch.setattr(module, "MAX_EXPANDED_BYTES", 1000)
    with pytest.raises(SkillError, match="expanded archive"):
        lifecycle.install_git(
            "check-output", "https://github.com/o/r", "a" * 40, ".", fetch=lambda _: data
        )
    assert list(lifecycle.root.iterdir()) == []


def test_update_conflict_rollback_and_removed_recovery(tmp_path):
    lifecycle = SkillLifecycle(tmp_path / "skills")
    first = lifecycle.install(package(tmp_path / "source"))
    changed = package(tmp_path / "replacement", "Check return status as well.")
    second = lifecycle.update("check-output", changed, expected=first["revision"])
    assert len(second["revisions"]) == 2
    with pytest.raises(SkillError, match="changed"):
        lifecycle.remove("check-output", expected=first["revision"])
    restored = lifecycle.rollback("check-output", first["revision"], expected=second["revision"])
    assert restored["revision"] == first["revision"]
    removed = lifecycle.remove("check-output", expected=restored["revision"])
    assert not removed["installed"] and len(removed["revisions"]) == 2
    recovered = lifecycle.rollback("check-output", second["revision"], expected=None)
    assert recovered["revision"] == second["revision"]
    (lifecycle.root / "check-output/SKILL.md").write_text(document("operator edit"))
    assert lifecycle.inspect("check-output")["modified"]
    with pytest.raises(SkillError, match="changed"):
        lifecycle.rollback("check-output", first["revision"], expected=second["revision"])
    assert "operator edit" in (lifecycle.root / "check-output/SKILL.md").read_text()


def test_identical_contents_keep_each_remote_pin_in_change_history(tmp_path):
    lifecycle = SkillLifecycle(tmp_path / "skills")
    data = archive([("repo/SKILL.md", document("same contents").encode(), None)])
    first = lifecycle.install_git(
        "check-output", "https://github.com/o/r", "a" * 40, ".", fetch=lambda _: data
    )
    second = lifecycle.install_git(
        "check-output",
        "https://github.com/o/r",
        "b" * 40,
        ".",
        expected=first["revision"],
        fetch=lambda _: data,
    )
    assert second["revision"] == first["revision"]
    assert len(second["revisions"]) == 1
    assert [change["provenance"]["commit"] for change in second["changes"]] == ["b" * 40, "a" * 40]


def test_interrupted_publication_recovers_from_durable_journal(tmp_path, monkeypatch):
    lifecycle = SkillLifecycle(tmp_path / "skills")
    original = lifecycle.install(package(tmp_path / "source"))
    replacement = package(tmp_path / "replacement", "New recovered body.")
    real_publish = lifecycle._publish

    def interrupted(name, files):
        real_publish(name, files)
        raise KeyboardInterrupt("process interrupted after filesystem publication")

    monkeypatch.setattr(lifecycle, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.update("check-output", replacement, expected=original["revision"])
    restarted = SkillLifecycle(lifecycle.root)
    state = restarted.inspect("check-output")
    assert not state["modified"] and state["revision"] != original["revision"]
    assert "New recovered body" in (lifecycle.root / "check-output/SKILL.md").read_text()


def sample_run(tmp_path, status: SessionStatus = "done", user=None):
    skill = package(tmp_path / "skills")
    library = SkillLibrary.load([skill.parent])
    session = Session(
        id="actual-run",
        provider="fake",
        model="fake",
        cwd=tmp_path,
        status=status,
        metadata={"active_skills": ["check-output"], "memory_scope": {"user_id": user}},
    )
    event = ActivityEvent(
        id="actual-event",
        session_id=session.id,
        kind="tool_call.completed",
        data={
            "name": "check",
            "duration_ms": 5,
            "is_error": status == "failed",
            "content_preview": "private output sentinel",
            "arguments": {"secret": "private argument sentinel"},
        },
    )
    return library, session, event


@pytest.mark.parametrize("status", ["done", "failed", "cancelled"])
def test_actual_evidence_is_deduplicated_private_text_is_hashed(tmp_path, status):
    library, session, event = sample_run(tmp_path, status)
    first = record_skill_run(session, [event], library)
    assert record_skill_run(session, [event], library) == first
    evidence = SkillLifecycle(tmp_path / "skills").evidence("check-output")
    assert len(evidence) == 1 and evidence[0]["status"] == status
    assert "private output sentinel" not in json.dumps(evidence)
    assert "private argument sentinel" not in json.dumps(evidence)
    assert evidence[0]["observations"][0]["event_id"] == "actual-event"
    assert "Check the output" in library.get("check-output").read()


def test_remote_paused_and_nonexecuted_calls_cannot_become_learning_evidence(tmp_path):
    library, session, event = sample_run(tmp_path, user="telegram:user")
    assert record_skill_run(session, [event], library) == []
    assert skill_evolution_tools(session, library) == []
    session.metadata["memory_scope"] = {"user_id": None}
    event.data["duration_ms"] = None
    assert record_skill_run(session, [event], library) == []
    event.data["duration_ms"] = 1
    session.status = "paused"
    assert record_skill_run(session, [event], library) == []


def test_proposal_requires_real_evidence_and_exact_reviewed_diff_and_owner(tmp_path):
    library, session, event = sample_run(tmp_path)
    lifecycle = SkillLifecycle(tmp_path / "skills")
    [evidence] = record_skill_run(session, [event], library)
    content = document("Verify return status before reporting success.")
    with pytest.raises(SkillError, match="recorded runs"):
        lifecycle.propose(
            "check-output",
            content,
            session_id="editor",
            evidence_ids=["invented"],
            reason="Improve checks",
        )
    proposal = lifecycle.propose(
        "check-output",
        content,
        session_id="editor",
        evidence_ids=[evidence],
        reason="Improve checks",
    )
    assert lifecycle.proposal(proposal["id"]) == proposal
    assert library.get("check-output").read() != content
    with pytest.raises(SkillError, match="exactly match"):
        lifecycle.apply(proposal["id"], expected=proposal["base"], reviewed_diff="abbreviated")
    with pytest.raises(SkillError, match="different session"):
        lifecycle.apply(
            proposal["id"],
            expected=proposal["base"],
            reviewed_diff=proposal["diff"],
            session_id="other",
        )
    lifecycle.apply(
        proposal["id"],
        expected=proposal["base"],
        reviewed_diff=proposal["diff"],
        session_id="editor",
    )
    with pytest.raises(SkillError, match="changed"):
        lifecycle.apply(
            proposal["id"],
            expected=proposal["base"],
            reviewed_diff=proposal["diff"],
            session_id="editor",
        )


async def test_real_run_proposal_pauses_until_approval_then_independent_session_reuses(tmp_path):
    directory = package(tmp_path / "skills")
    library = SkillLibrary.load([directory.parent])
    store = InMemoryStorage()
    registry = ToolRegistry()
    registry.register(MockTool(name="check"))
    adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(call_id="read", name="skill_read", arguments={"name": "check-output"}),
            tool_call_turn(call_id="checked", name="check", arguments={"text": "result status=0"}),
            text_turn("Checked successfully."),
        ],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        activity_store=store,
        failover=FailoverPolicy(chain=["mock"]),
        skill_library=library,
        default_cwd=str(tmp_path),
    )
    async for _ in agent.run(RunRequest(prompt="Use check-output", model="m", session_id="first")):
        pass
    first = await store.get("first")
    assert first is not None and first.status == "done"
    identifiers = record_skill_run(first, await store.list_activity(session_id="first"), library)
    assert len(identifiers) == 1
    editor = Session(id="editor", provider="mock", model="m", cwd=tmp_path)
    tools = {tool.name: tool for tool in skill_evolution_tools(editor, library)}
    proposed = await tools["skill_propose"](
        ToolCall(
            id="propose",
            name="skill_propose",
            arguments={
                "name": "check-output",
                "content": document("Verify return status before reporting success."),
                "evidence_ids": identifiers,
                "reason": "Preserve the successful check procedure.",
            },
        )
    )
    assert not proposed.is_error
    proposal = json.loads(proposed.content)
    args = {
        "name": "check-output",
        "proposal_id": proposal["id"],
        "expected_base": proposal["base"],
        "reviewed_diff": proposal["diff"],
    }
    adapter.scripts = [
        tool_call_turn(call_id="apply", name="skill_apply", arguments=args),
        text_turn("Applied reviewed improvement."),
    ]
    agent.approval_store = store
    agent.approval_handler = InboxApprovalHandler(approval_store=store)
    agent.pause_on_approval = True
    async for _ in agent.run(
        RunRequest(prompt="Apply the proposed improvement", model="m", session_id="editor")
    ):
        pass
    [pending] = await store.list_approvals(status="pending")
    assert pending.arguments["reviewed_diff"] == proposal["diff"]
    assert "Verify return status" not in library.get("check-output").read()
    await store.resolve_approval(pending.id, status="granted")
    async for _ in agent.resume("editor", prompt="Continue the approved change"):
        pass
    assert "Verify return status" in library.get("check-output").read()
    later_adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(
                call_id="later-read", name="skill_read", arguments={"name": "check-output"}
            ),
            text_turn("I will verify return status."),
        ],
    )
    fresh = Agent(
        adapters={"mock": later_adapter},
        tools=ToolRegistry(),
        storage=InMemoryStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        skill_library=SkillLibrary.load([directory.parent]),
        default_cwd=str(tmp_path),
    )
    async for _ in fresh.run(
        RunRequest(prompt="Use check-output", model="m", session_id="independent")
    ):
        pass
    assert any(
        "Verify return status" in (message.content or "")
        for message in later_adapter.calls[-1]["messages"]
    )
    assert not any(
        "Checked successfully" in (message.content or "")
        for message in later_adapter.calls[-1]["messages"]
    )


async def test_empty_library_learns_new_skill_from_real_run_only_after_review(tmp_path):
    library = SkillLibrary()
    store = InMemoryStorage()
    registry = ToolRegistry()
    registry.register(MockTool(name="check"))
    adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(
                call_id="actual-check", name="check", arguments={"text": "private actual output"}
            ),
            text_turn("Work completed."),
        ],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        activity_store=store,
        approval_store=store,
        approval_handler=InboxApprovalHandler(approval_store=store),
        pause_on_approval=True,
        failover=FailoverPolicy(chain=["mock"]),
        skill_library=library,
        default_cwd=str(tmp_path),
    )
    async for _ in agent.run(
        RunRequest(prompt="Complete useful work", model="m", session_id="work")
    ):
        pass
    lifecycle = SkillLifecycle(tmp_path / ".harness/skills")
    [evidence] = lifecycle.evidence()
    assert evidence["session_id"] == "work" and evidence["skill"] is None
    assert "private actual output" not in json.dumps(evidence)
    assert not library.skills
    adapter.scripts = [
        tool_call_turn(call_id="evidence", name="skill_evidence", arguments={}),
        tool_call_turn(
            call_id="proposal",
            name="skill_propose",
            arguments={
                "name": "check-output",
                "content": document("Inspect the actual tool result before claiming success."),
                "reason": "Reuse the completed checking procedure.",
                "evidence_ids": [evidence["id"]],
                "create": True,
            },
        ),
        text_turn("A new skill is proposed for review."),
    ]
    async for _ in agent.run(
        RunRequest(
            prompt="Propose a reusable skill from that work", model="m", session_id="learner"
        )
    ):
        pass
    learner = await store.get("learner")
    assert learner is not None
    proposal = json.loads(
        next(
            item.content
            for item in learner.messages
            if item.role == "tool" and item.name == "skill_propose"
        )
        or "{}"
    )
    assert (
        proposal["create"] and proposal["base"] == "absent" and "--- /dev/null" in proposal["diff"]
    )
    assert not (lifecycle.root / "check-output").exists()
    adapter.scripts = [
        tool_call_turn(
            call_id="apply-new",
            name="skill_apply",
            arguments={
                "name": "check-output",
                "proposal_id": proposal["id"],
                "expected_base": "absent",
                "reviewed_diff": proposal["diff"],
            },
        ),
        text_turn("Created the approved skill."),
    ]
    async for _ in agent.resume("learner", prompt="Apply the proposed new skill"):
        pass
    [pending] = await store.list_approvals(status="pending")
    assert (
        pending.tool_name == "skill_apply"
        and pending.arguments["reviewed_diff"] == proposal["diff"]
    )
    assert not (lifecycle.root / "check-output").exists()
    await store.resolve_approval(pending.id, status="granted")
    async for _ in agent.resume("learner", prompt="Continue after review"):
        pass
    assert library.get("check-output").directory == lifecycle.root / "check-output"
    assert lifecycle.inspect("check-output")["changes"][0]["provenance"]["kind"] == "creation"
    later_adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(
                call_id="read-new", name="skill_read", arguments={"name": "check-output"}
            ),
            text_turn("Following the new procedure."),
        ],
    )
    later = Agent(
        adapters={"mock": later_adapter},
        tools=ToolRegistry(),
        storage=InMemoryStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        skill_library=SkillLibrary.load([lifecycle.root]),
        default_cwd=str(tmp_path),
    )
    async for _ in later.run(
        RunRequest(prompt="Use check-output", model="m", session_id="independent")
    ):
        pass
    assert any(
        "Inspect the actual tool result" in (item.content or "")
        for item in later_adapter.calls[-1]["messages"]
    )
    assert not any(
        "private actual output" in (item.content or "")
        for item in later_adapter.calls[-1]["messages"]
    )


def test_new_skill_creation_conflict_and_failed_generic_evidence(tmp_path):
    session = Session(
        id="failed-real-run", provider="fake", model="fake", cwd=tmp_path, status="failed"
    )
    event = ActivityEvent(
        session_id=session.id,
        kind="tool_call.completed",
        data={"name": "check", "duration_ms": 1, "is_error": True},
    )
    [identifier] = record_skill_run(session, [event], SkillLibrary())
    lifecycle = SkillLifecycle(tmp_path / ".harness/skills")
    assert lifecycle.evidence()[0]["status"] == "failed"
    assert not (lifecycle.root / "check-output").exists()
    proposed = lifecycle.propose(
        "check-output",
        document("Inspect failure output."),
        session_id="editor",
        evidence_ids=[identifier],
        reason="Record a lesson from failure",
        create=True,
    )
    package(lifecycle.root, "An operator installed this while review was pending.")
    with pytest.raises(SkillError, match="appeared"):
        lifecycle.apply(
            proposed["id"], expected="absent", reviewed_diff=proposed["diff"], session_id="editor"
        )
    assert "An operator installed" in (lifecycle.root / "check-output/SKILL.md").read_text()
