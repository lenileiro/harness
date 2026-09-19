from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from harness.core import (
    Agent,
    ApprovalPolicy,
    AutoApprove,
    Done,
    FailoverPolicy,
    ToolRegistry,
    Usage,
)
from harness.core.adapter import Adapter
from harness.core.mission.execution import execute_mission_agents
from harness.core.mission.runtime import execute_mission_burst

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn
from .test_mission_runtime import _seed_two_milestone_mission


def _fixture(tmp_path: Path, *, verification: str | None = None):
    store, mission_id = _seed_two_milestone_mission(tmp_path)
    contract = store.load_contract_for_mission(mission_id)
    features = {f.id: f for f in store.list_features(mission_id=mission_id)}
    assertions = []
    for assertion in contract.assertions:
        target = features[assertion.covered_by_features[0]].target_files[0]
        command = (
            verification
            or f"from pathlib import Path; assert Path({target!r}).read_text() == 'implemented'"
        )
        assertions.append(replace(assertion, command=(sys.executable, "-c", command)))
    store.add_contract(replace(contract, assertions=tuple(assertions)))
    return store, mission_id


def _factory(tmp_path: Path, *, storage=None, adapters=None):
    storage = storage or MockStorage()
    adapters = adapters if adapters is not None else []

    def factory(mission, feature):
        def write_target(**kwargs):
            target = tmp_path / feature.target_files[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("implemented", encoding="utf-8")
            return "Created target file"

        scripts = [
            tool_call_turn(call_id="write", name="create_target", arguments={"text": "implement"}),
            text_turn("Implemented feature"),
        ]
        for script in scripts:
            for event in script:
                if isinstance(event, Done):
                    event.usage = Usage(total_tokens=100, prompt_tokens=60, completion_tokens=40)
        adapter = MockAdapter("mock", scripts=scripts)
        adapters.append(adapter)
        tools = ToolRegistry()
        tools.register(MockTool(name="create_target", responder=write_target))
        return Agent(
            adapters={"mock": adapter},
            tools=tools,
            storage=storage,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            approval_policy=ApprovalPolicy(default="auto"),
            approval_handler=AutoApprove(),
            default_model="test-model",
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

    return factory


async def test_real_workers_create_targets_and_run_independent_assertions(tmp_path: Path) -> None:
    store, mission_id = _fixture(tmp_path)
    result = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=_factory(tmp_path),
    )
    assert result.status == "completed"
    assert (tmp_path / "app/one.py").read_text() == "implemented"
    assert (tmp_path / "app/two.py").read_text() == "implemented"
    mission = store.load_mission(mission_id)
    assert mission.execution_mode == "agent"
    assert mission.tokens_used == 400
    assert all(
        f.status == "validated" and f.worker_session_id
        for f in store.list_features(mission_id=mission_id)
    )
    evidence = [
        json.loads(path.read_text()) for path in store.runs_dir.glob("verification-*/*/result.json")
    ]
    assert len(evidence) == 2
    assert all(item["status"] == "passed" and item["exit_code"] == 0 for item in evidence)


async def test_worker_claim_cannot_override_failed_assertion(tmp_path: Path) -> None:
    store, mission_id = _fixture(tmp_path, verification="raise SystemExit(7)")
    result = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=_factory(tmp_path),
    )
    assert result.status == "blocked"
    assert result.stop_reason == "verification_failed"
    assert store.load_mission(mission_id).status == "blocked"
    assert not any(f.status == "validated" for f in store.list_features(mission_id=mission_id))
    evidence = [
        json.loads(path.read_text()) for path in store.runs_dir.glob("verification-*/*/result.json")
    ]
    assert evidence[0]["exit_code"] == 7
    with pytest.raises(ValueError, match="cannot be auto-completed"):
        execute_mission_burst(store=store, mission_id=mission_id, auto_complete=True)


async def test_missing_assertion_commands_rejects_before_worker(tmp_path: Path) -> None:
    store, mission_id = _seed_two_milestone_mission(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="explicit argv command"):
        await execute_mission_agents(
            store=store,
            mission_id=mission_id,
            cwd=tmp_path,
            agent_factory=_factory(tmp_path, adapters=calls),
        )
    assert calls == []
    assert all(f.status == "pending" for f in store.list_features(mission_id=mission_id))


@pytest.mark.parametrize("coverage", ["cross_milestone", "unknown_feature", "empty"])
async def test_invalid_assertion_coverage_rejects_before_state_changes(
    tmp_path: Path,
    coverage: str,
) -> None:
    store, mission_id = _fixture(tmp_path)
    contract = store.load_contract_for_mission(mission_id)
    original_mission = store.load_mission(mission_id)
    original_features = store.list_features(mission_id=mission_id)
    if coverage == "cross_milestone":
        feature_ids = tuple(feature.id for feature in original_features)
        reason = "split into milestone-local assertions"
    elif coverage == "unknown_feature":
        feature_ids = ("missing-feature",)
        reason = "unknown mission feature IDs"
    else:
        feature_ids = ()
        reason = "at least one mission feature"
    store.add_contract(
        replace(
            contract,
            assertions=(
                replace(contract.assertions[0], covered_by_features=feature_ids),
                *contract.assertions[1:],
            ),
        )
    )
    workers = []
    with pytest.raises(ValueError, match=reason):
        await execute_mission_agents(
            store=store,
            mission_id=mission_id,
            cwd=tmp_path,
            agent_factory=_factory(tmp_path, adapters=workers),
        )
    assert workers == []
    assert store.load_mission(mission_id) == original_mission
    assert store.list_features(mission_id=mission_id) == original_features
    assert not (store.root / "execution-locks").exists()


async def test_interrupted_worker_resumes_saved_session(tmp_path: Path) -> None:
    store, mission_id = _fixture(tmp_path)
    storage = MockStorage()

    class SlowAdapter(MockAdapter):
        async def _stream(self):
            await asyncio.sleep(2)
            for event in text_turn("late"):
                yield event

    def slow_factory(mission, feature):
        agent = _factory(tmp_path, storage=storage)(mission, feature)
        agent.adapters = {"mock": cast(Adapter, SlowAdapter("mock"))}
        return agent

    first = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=slow_factory,
        timeout_seconds=0.03,
    )
    assert first.status == "paused"
    feature = next(f for f in store.list_features(mission_id=mission_id) if f.status == "blocked")
    session_id = feature.worker_session_id
    assert session_id
    assert await storage.get(session_id) is not None
    handoffs = store.list_handoffs(mission_id=mission_id, feature_id=feature.id)
    assert any(session_id in h.next_recommendation for h in handoffs)
    resumed = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=_factory(tmp_path, storage=storage),
    )
    assert resumed.status == "completed"
    assert store.load_feature(feature.id).worker_session_id == session_id


async def test_assertion_timeout_persists_failure_evidence(tmp_path: Path) -> None:
    store, mission_id = _fixture(tmp_path, verification="import time; time.sleep(2)")
    contract = store.load_contract_for_mission(mission_id)
    store.add_contract(
        replace(
            contract,
            assertions=tuple(replace(a, timeout_seconds=0.03) for a in contract.assertions),
        )
    )
    result = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=_factory(tmp_path),
    )
    assert result.status == "blocked"
    evidence = [
        json.loads(path.read_text()) for path in store.runs_dir.glob("verification-*/*/result.json")
    ]
    assert evidence[0]["timed_out"] is True


async def test_token_budget_blocks_provider_requests_and_persists_state(tmp_path: Path) -> None:
    store, mission_id = _fixture(tmp_path)
    store.update_mission(replace(store.load_mission(mission_id), budget_tokens=1))
    adapters = []
    result = await execute_mission_agents(
        store=store,
        mission_id=mission_id,
        cwd=tmp_path,
        agent_factory=_factory(tmp_path, adapters=adapters),
    )
    assert result.status == "blocked"
    assert len(adapters) == 1
    assert adapters[0].calls == []
    assert store.load_mission(mission_id).status == "blocked"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group cleanup")
def test_assertion_success_stops_background_descendants(tmp_path: Path) -> None:
    import os
    import signal
    import time
    from contextlib import suppress

    from harness.core.mission.validator import run_assertion_command

    store, mission_id = _fixture(tmp_path)
    assertion = store.load_contract_for_mission(mission_id).assertions[0]
    child_code = (
        "from pathlib import Path; import time; "
        "Path('child-ready').write_text('ready'); "
        "time.sleep(0.25); Path('late-write').write_text('escaped'); time.sleep(30)"
    )
    parent_code = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "Path('child-pid').write_text(str(child.pid)); "
        "\nwhile not Path('child-ready').exists(): time.sleep(0.005)\n"
    )
    assertion = replace(assertion, command=(sys.executable, "-c", parent_code))
    try:
        result = run_assertion_command(
            assertion=assertion, cwd=tmp_path, evidence_dir=tmp_path / "evidence"
        )
        assert result["status"] == "passed"
        assert result["exit_code"] == 0
        time.sleep(0.4)
        assert not (tmp_path / "late-write").exists()
    finally:
        pid_file = tmp_path / "child-pid"
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group cleanup")
def test_assertion_interrupt_kills_reaps_and_persists_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from harness.core.mission.validator import run_assertion_command

    store, mission_id = _fixture(tmp_path)
    assertion = replace(
        store.load_contract_for_mission(mission_id).assertions[0],
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
    )
    original_wait = subprocess.Popen.wait
    interrupted = []

    def wait(process, timeout=None):
        if not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_assertion_command(
                assertion=assertion, cwd=tmp_path, evidence_dir=tmp_path / "evidence"
            )
        assert len(interrupted) == 1
        assert interrupted[0].returncode is not None
        evidence = json.loads((tmp_path / "evidence/result.json").read_text())
        assert evidence["status"] == "failed"
        assert evidence["interrupted"] is True
        assert "KeyboardInterrupt" in evidence["reason"]
        assert evidence["exit_code"] is not None
    finally:
        for process in interrupted:
            if process.poll() is None:
                process.kill()
            original_wait(process, timeout=2)
