from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from harness.cli.common import (
    _build_adapter,
    _build_tools,
    _load_cli_config,
    _resolve_chain,
    _run_async,
    console,
)
from harness.cli.config import default_config_path, load_config
from harness.cli.run_commands import run_once as _run_once_impl
from harness.cli.runtime_agent import build_agent as _build_agent_impl
from harness.cli.runtime_helpers import (
    build_critic as _build_critic,
)
from harness.cli.runtime_helpers import (
    build_search_fn as _build_search_fn,
)
from harness.cli.runtime_helpers import (
    build_storage as _build_storage,
)
from harness.cli.runtime_helpers import (
    build_verifier as _build_verifier,
)
from harness.cli.runtime_helpers import (
    print_defense_ledger as _print_defense_ledger,
)
from harness.cli.runtime_helpers import (
    resolve_runtime_strategy as _resolve_runtime_strategy,
)
from harness.core import (
    Mission,
    MissionLoopStep,
    MissionStore,
    MissionSummaryReport,
    build_mission_summary_report,
    complete_mission_feature,
    default_mission_root,
    execute_mission_burst,
    execute_mission_milestone,
    execute_next_mission_feature,
    list_mission_reports,
    load_mission_report,
    validate_mission_milestone,
    write_mission_scheduled_run_record,
    write_mission_summary_report,
)
from harness.core.mission_planner import (
    PlannedAssertionInput,
    PlannedFeatureInput,
    PlannedMilestoneInput,
    build_mission_plan,
    parse_mission_plan_draft,
)
from harness.core.opportunities import Opportunity
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research_store import ResearchStore, default_research_root

mission_app = typer.Typer(
    name="mission",
    help="Plan and track multi-step autonomous missions.",
    no_args_is_help=True,
)

_MISSION_PLANNER_SYSTEM_PROMPT = (
    "You are a mission planning agent. Turn a high-level software mission goal into a "
    "bounded delivery plan before implementation starts. Return JSON only."
)


def _build_agent(
    *,
    chain: list[str],
    base_url: str | None,
    model: str,
    storage: Any,
    cwd: Path,
    config: Any,
    yes: bool,
    inbox: bool = False,
    activity_store: Any = None,
    approval_store: Any = None,
    verifier: Any = None,
    critic: Any = None,
    budget: Any = None,
    memory_store: Any | None = None,
    planner: Any | None = None,
    session_overrides: dict[str, Any] | None = None,
    predictor: Any | None = None,
    repair: Any | None = None,
    system_prompt: str | None = None,
    compactor: Any | None = None,
    max_repair_attempts: int = 3,
    profile: str = "minimal",
    phases_enabled: bool = False,
    loop_detector: Any | None = None,
    contracts: Any | None = None,
    tips_provider: Any | None = None,
    resume: Any | None = None,
    build_tools: Any = _build_tools,
) -> Any:
    return _build_agent_impl(
        chain=chain,
        base_url=base_url,
        model=model,
        storage=storage,
        cwd=cwd,
        config=config,
        yes=yes,
        build_adapter=_build_adapter,
        build_tools=build_tools,
        build_search_fn=_build_search_fn,
        console=console,
        inbox=inbox,
        activity_store=activity_store,
        approval_store=approval_store,
        verifier=verifier,
        critic=critic,
        budget=budget,
        memory_store=memory_store,
        planner=planner,
        session_overrides=session_overrides,
        predictor=predictor,
        repair=repair,
        system_prompt=system_prompt,
        compactor=compactor,
        max_repair_attempts=max_repair_attempts,
        profile=profile,
        phases_enabled=phases_enabled,
        loop_detector=loop_detector,
        contracts=contracts,
        tips_provider=tips_provider,
        resume=resume,
    )


async def _resolve_task_attachment(
    storage: object, task_ref: str | None, session_id: str | None
) -> tuple[str | None, None]:
    del storage, task_ref, session_id
    return None, None


def _render_noop(event: Any) -> None:
    del event


def build_mission_plan_prompt(*, mission: Mission) -> str:
    return (
        f"Create a mission plan for '{mission.title}'.\n\n"
        f"Goal:\n{mission.goal.strip()}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "contract_summary": "string",\n'
        '  "milestones": [{"label": "m1", "title": "string", "summary": "string"}],\n'
        '  "assertions": [{"label": "a1", "title": "string", "description": "string", '
        '"kind": "contract", "verification_method": "string"}],\n'
        '  "features": [{"label": "f1", "milestone_label": "m1", "title": "string", '
        '"summary": "string", "assigned_role": "worker", "target_files": ["path"], '
        '"depends_on_labels": [], "assertion_labels": ["a1"], "research_refs": []}]\n'
        "}\n\n"
        "Rules:\n"
        "- At least one milestone, assertion, and feature.\n"
        "- Every feature must cover one or more assertions.\n"
        "- Keep the plan bounded and implementation-oriented.\n"
        "- Use labels like m1, a1, f1 for stable references.\n"
        "- Prefer assigned_role='worker' unless a different role is necessary.\n"
    )


def _generate_mission_plan_text(
    *,
    mission: Mission,
    model: str | None,
    provider: str | None,
    failover: str | None,
    base_url: str | None,
    cwd: Path,
    max_steps: int,
    max_output_tokens: int | None,
    config_path: Path | None,
) -> str | None:
    fixture_path = os.environ.get("HARNESS_MISSION_PLAN_DRAFT_FILE")
    if fixture_path:
        return Path(fixture_path).read_text(encoding="utf-8")
    cfg = _load_cli_config(config_path)
    chain = _resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"
    prompt = build_mission_plan_prompt(mission=mission)
    return _run_async(
        _run_once_impl(
            prompt=prompt,
            model=effective_model,
            chain=chain,
            base_url=base_url,
            cwd=cwd,
            max_steps=max_steps,
            max_output_tokens=max_output_tokens,
            session_id=None,
            task_ref=None,
            db=None,
            in_memory=True,
            yes=True,
            inbox=False,
            verify="none",
            verify_command=None,
            critic=None,
            require_tools=True,
            goal=False,
            max_context_tokens=None,
            predict=False,
            auto_compact=False,
            max_repair=1,
            profile="bare",
            domain="mission-planning",
            phases=None,
            loop_detect=False,
            contracts=False,
            tips=True,
            silent=True,
            config=cfg,
            build_storage=_build_storage,
            resolve_task_attachment=_resolve_task_attachment,
            resolve_runtime_strategy=_resolve_runtime_strategy,
            build_verifier=_build_verifier,
            build_critic=_build_critic,
            build_adapter=_build_adapter,
            build_tools=_build_tools,
            build_agent=_build_agent,
            print_defense_ledger=_print_defense_ledger,
            render=_render_noop,
            default_system_prompt=_MISSION_PLANNER_SYSTEM_PROMPT,
            console=console,
        )
    )


@mission_app.command("create")
def mission_create_command(
    *,
    title: str = typer.Option(..., "--title"),
    goal: str = typer.Option(..., "--goal"),
    created_by: str = typer.Option("human", "--created-by"),
    planner_model: str = typer.Option("", "--planner-model"),
    worker_model: str = typer.Option("", "--worker-model"),
    validator_model: str = typer.Option("", "--validator-model"),
    reporter_model: str = typer.Option("", "--reporter-model"),
    planner_brief: str = typer.Option("", "--planner-brief"),
    worker_brief: str = typer.Option("", "--worker-brief"),
    validator_brief: str = typer.Option("", "--validator-brief"),
    reporter_brief: str = typer.Option("", "--reporter-brief"),
    budget_tokens: int | None = typer.Option(None, "--budget-tokens"),
    budget_runtime_minutes: int | None = typer.Option(None, "--budget-runtime-minutes"),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    cfg = load_config(config_path)
    mission_roles = cfg.mission_roles
    store = MissionStore(root=default_mission_root(working_dir))
    mission = Mission(
        id=store.new_id("mission", title),
        title=title.strip(),
        goal=goal.strip(),
        created_by=created_by.strip() or "human",
        planner_model=planner_model.strip() or (mission_roles.planner.model or ""),
        worker_model=worker_model.strip() or (mission_roles.worker.model or ""),
        validator_model=validator_model.strip() or (mission_roles.validator.model or ""),
        reporter_model=reporter_model.strip() or (mission_roles.reporter.model or ""),
        planner_brief=planner_brief.strip() or (mission_roles.planner.brief or ""),
        worker_brief=worker_brief.strip() or (mission_roles.worker.brief or ""),
        validator_brief=validator_brief.strip() or (mission_roles.validator.brief or ""),
        reporter_brief=reporter_brief.strip() or (mission_roles.reporter.brief or ""),
        budget_tokens=budget_tokens,
        budget_runtime_minutes=budget_runtime_minutes,
    )
    target = store.add_mission(mission)
    console.print(f"[green]Created mission {mission.id}[/green] at {target}")


@mission_app.command("show")
def mission_show_command(
    mission_id: str = typer.Argument(..., help="Mission id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        mission = store.load_mission(mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    console.print(f"[bold]{mission.title}[/bold]")
    console.print(f"id={mission.id}")
    console.print(f"status={mission.status}")
    console.print(f"created_by={mission.created_by}")
    if mission.current_milestone_id:
        console.print(f"current_milestone_id={mission.current_milestone_id}")
    if mission.budget_tokens is not None:
        console.print(f"budget_tokens={mission.budget_tokens}")
    if mission.budget_runtime_minutes is not None:
        console.print(f"budget_runtime_minutes={mission.budget_runtime_minutes}")
    role_models = {
        "planner": mission.planner_model,
        "worker": mission.worker_model,
        "validator": mission.validator_model,
        "reporter": mission.reporter_model,
    }
    role_briefs = {
        "planner": mission.planner_brief,
        "worker": mission.worker_brief,
        "validator": mission.validator_brief,
        "reporter": mission.reporter_brief,
    }
    if any(role_models.values()) or any(role_briefs.values()):
        console.print("\n[bold]Role Profiles[/bold]")
        for role in ("planner", "worker", "validator", "reporter"):
            if role_models[role]:
                console.print(f"- {role}_model={role_models[role]}")
            if role_briefs[role]:
                console.print(f"  brief={role_briefs[role]}")
    console.print("\n[bold]Goal[/bold]")
    console.print(mission.goal)


@mission_app.command("list")
def mission_list_command(
    *,
    status: str | None = typer.Option(None, "--status"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    missions = store.list_missions(status=status)
    if not missions:
        console.print("[dim]No missions found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Status", no_wrap=True)
    table.add_column("Created By", no_wrap=True)
    table.add_column("Current Milestone", no_wrap=True)
    for mission in missions:
        table.add_row(
            mission.id,
            mission.title,
            mission.status,
            mission.created_by,
            mission.current_milestone_id or "—",
        )
    console.print(table)


def _emit_json(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def _print_mission_steps(steps: tuple[MissionLoopStep, ...]) -> None:
    for index, step in enumerate(steps, start=1):
        console.print(f"{index}. [{step.kind}] {step.status} {step.message}")
        if step.milestone_id:
            console.print(f"   milestone_id={step.milestone_id}")
        if step.feature_id:
            console.print(f"   feature_id={step.feature_id}")
        if step.run_id:
            console.print(f"   run_id={step.run_id}")
        if step.handoff_id:
            console.print(f"   handoff_id={step.handoff_id}")


def _print_mission_report(report: MissionSummaryReport) -> None:
    console.print(f"[bold]{report.id}[/bold]")
    console.print(f"mission_id={report.mission_id}")
    console.print(f"status={report.status}")
    if report.current_milestone_id:
        console.print(f"current_milestone_id={report.current_milestone_id}")
    console.print("\n[bold]Summary[/bold]")
    console.print(report.summary)
    if report.role_profiles:
        console.print("\n[bold]Role Profiles[/bold]")
        for item in report.role_profiles:
            suffix = f" model={item['model']}" if item["model"] else ""
            console.print(f"- {item['role']}{suffix}")
            if item["brief"]:
                console.print(f"  brief={item['brief']}")
    if report.next_actions:
        console.print("\n[bold]Next Actions[/bold]")
        for item in report.next_actions:
            console.print(f"- {item}")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_milestone_spec(spec: str) -> PlannedMilestoneInput:
    parts = [part.strip() for part in spec.split("|", 2)]
    if len(parts) != 3 or any(not part for part in parts):
        raise typer.BadParameter("--milestone must be 'label|title|summary'")
    return PlannedMilestoneInput(label=parts[0], title=parts[1], summary=parts[2])


def _parse_assertion_spec(spec: str) -> PlannedAssertionInput:
    parts = [part.strip() for part in spec.split("|", 4)]
    if len(parts) != 5 or any(not part for part in parts):
        raise typer.BadParameter(
            "--assertion must be 'label|title|description|kind|verification_method'"
        )
    return PlannedAssertionInput(
        label=parts[0],
        title=parts[1],
        description=parts[2],
        kind=parts[3],
        verification_method=parts[4],
    )


def _parse_feature_spec(spec: str) -> PlannedFeatureInput:
    parts = [part.strip() for part in spec.split("|", 8)]
    if len(parts) not in {8, 9}:
        raise typer.BadParameter(
            "--feature must be "
            "'label|milestone_label|title|summary|assigned_role|target_files_csv|depends_on_csv|assertion_labels_csv[|research_refs_csv]'"
        )
    if len(parts) == 8:
        parts.append("")
    (
        label,
        milestone_label,
        title,
        summary,
        assigned_role,
        target_files,
        depends_on,
        assertions,
        research_refs,
    ) = parts
    if not label or not milestone_label or not title or not summary:
        raise typer.BadParameter(
            "--feature label, milestone_label, title, and summary are required"
        )
    return PlannedFeatureInput(
        label=label,
        milestone_label=milestone_label,
        title=title,
        summary=summary,
        assigned_role=assigned_role or "worker",
        target_files=_split_csv(target_files),
        depends_on_labels=_split_csv(depends_on),
        assertion_labels=_split_csv(assertions),
        research_refs=_split_csv(research_refs),
    )


@mission_app.command("plan")
def mission_plan_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    contract_summary: str = typer.Option(..., "--contract-summary"),
    milestone: list[str] = typer.Option([], "--milestone"),
    assertion: list[str] = typer.Option([], "--assertion"),
    feature: list[str] = typer.Option([], "--feature"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        mission = store.load_mission(mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    try:
        plan = build_mission_plan(
            store=store,
            mission=mission,
            contract_summary=contract_summary,
            milestones=tuple(_parse_milestone_spec(item) for item in milestone),
            assertions=tuple(_parse_assertion_spec(item) for item in assertion),
            features=tuple(_parse_feature_spec(item) for item in feature),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    store.update_mission(plan.mission)
    for item in plan.milestones:
        store.add_milestone(item)
    for item in plan.features:
        store.add_feature(item)
    store.add_contract(plan.contract)
    console.print(f"[green]Planned mission {plan.mission.id}[/green]")
    console.print(f"milestones={len(plan.milestones)}")
    console.print(f"features={len(plan.features)}")
    console.print(f"assertions={len(plan.contract.assertions)}")
    console.print(f"current_milestone_id={plan.mission.current_milestone_id}")


@mission_app.command("draft-plan")
def mission_draft_plan_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    apply: bool = typer.Option(False, "--apply"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider"),
    failover: str | None = typer.Option(None, "--failover"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_steps: int = typer.Option(20, "--max-steps"),
    max_output_tokens: int | None = typer.Option(None, "--max-output-tokens"),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        mission = store.load_mission(mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc

    if apply and (
        store.list_milestones(mission_id=mission_id)
        or store.list_features(mission_id=mission_id)
        or store.list_contracts(mission_id=mission_id)
    ):
        raise typer.BadParameter("mission draft-plan --apply requires an unplanned mission")

    raw_text = _generate_mission_plan_text(
        mission=mission,
        model=model,
        provider=provider,
        failover=failover,
        base_url=base_url,
        cwd=working_dir,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        config_path=config_path,
    )
    parsed = parse_mission_plan_draft(raw_text or "")
    if parsed is None:
        if json_output:
            _emit_json({"raw_output": raw_text or ""})
            return
        console.print(raw_text or "")
        raise typer.Exit(1)

    payload: dict[str, object] = {"draft": parsed.to_dict()}
    if apply:
        try:
            plan = build_mission_plan(
                store=store,
                mission=mission,
                contract_summary=parsed.contract_summary,
                milestones=parsed.milestones,
                assertions=parsed.assertions,
                features=parsed.features,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        store.update_mission(plan.mission)
        for item in plan.milestones:
            store.add_milestone(item)
        for item in plan.features:
            store.add_feature(item)
        store.add_contract(plan.contract)
        payload["applied"] = {
            "mission_id": plan.mission.id,
            "milestones": len(plan.milestones),
            "features": len(plan.features),
            "assertions": len(plan.contract.assertions),
            "current_milestone_id": plan.mission.current_milestone_id,
        }

    if json_output:
        _emit_json(payload)
        return
    console.print(f"[green]Drafted mission plan for {mission_id}[/green]")
    console.print(f"milestones={len(parsed.milestones)}")
    console.print(f"features={len(parsed.features)}")
    console.print(f"assertions={len(parsed.assertions)}")
    if apply and "applied" in payload:
        applied = payload["applied"]
        assert isinstance(applied, dict)
        console.print("[green]Applied drafted plan[/green]")
        console.print(f"current_milestone_id={applied['current_milestone_id']}")


@mission_app.command("approve")
def mission_approve_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        mission = store.load_mission(mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    milestones = store.list_milestones(mission_id=mission_id)
    features = store.list_features(mission_id=mission_id)
    contracts = store.list_contracts(mission_id=mission_id)
    if not milestones or not features or not contracts:
        raise typer.BadParameter(
            "mission approval requires a plan with milestones, features, and a validation contract"
        )
    approved = Mission.from_dict({**mission.to_dict(), "status": "approved"})
    store.update_mission(approved)
    console.print(f"[green]Approved mission {mission_id}[/green]")


@mission_app.command("list-milestones")
def mission_list_milestones_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    milestones = store.list_milestones(mission_id=mission_id)
    if not milestones:
        console.print("[dim]No milestones found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Status", no_wrap=True)
    table.add_column("Order", no_wrap=True)
    for item in milestones:
        table.add_row(item.id, item.title, item.status, str(item.order))
    console.print(table)


@mission_app.command("list-features")
def mission_list_features_command(
    *,
    mission_id: str | None = typer.Option(None, "--mission"),
    milestone_id: str | None = typer.Option(None, "--milestone"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    features = store.list_features(mission_id=mission_id, milestone_id=milestone_id)
    if not features:
        console.print("[dim]No features found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Milestone", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Research Refs", overflow="fold")
    for item in features:
        table.add_row(
            item.id,
            item.title,
            item.milestone_id,
            item.status,
            item.assigned_role,
            ", ".join(item.research_refs) or "—",
        )
    console.print(table)


@mission_app.command("show-contract")
def mission_show_contract_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        contract = store.load_contract_for_mission(mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no validation contract found for mission: {mission_id!r}"
        ) from exc
    console.print(f"[bold]{contract.id}[/bold]")
    console.print(f"mission_id={contract.mission_id}")
    console.print("\n[bold]Summary[/bold]")
    console.print(contract.summary)
    if contract.assertions:
        console.print("\n[bold]Assertions[/bold]")
        for assertion in contract.assertions:
            console.print(
                f"- {assertion.id}: {assertion.title} [{assertion.kind}] -> "
                f"{assertion.verification_method}"
            )


@mission_app.command("execute-next")
def mission_execute_next_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = execute_next_mission_feature(store=store, mission_id=mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        _emit_json(result.to_dict())
        return
    color = {
        "dispatched": "green",
        "completed": "green",
        "no_work": "dim",
        "blocked": "yellow",
        "running": "blue",
    }.get(result.status, "white")
    console.print(f"[{color}]{result.status}[/{color}] {result.message}")
    if result.milestone_id:
        console.print(f"milestone_id={result.milestone_id}")
    if result.feature_id:
        console.print(f"feature_id={result.feature_id}")
    if result.run_id:
        console.print(f"run_id={result.run_id}")
    if result.handoff_id:
        console.print(f"handoff_id={result.handoff_id}")


@mission_app.command("complete-feature")
def mission_complete_feature_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    feature_id: str = typer.Option(..., "--feature"),
    completed_work: str = typer.Option(..., "--completed-work"),
    remaining_work: str = typer.Option("", "--remaining-work"),
    known_issue: list[str] = typer.Option([], "--known-issue"),
    next_recommendation: str = typer.Option("", "--next-recommendation"),
    confidence: float = typer.Option(0.9, "--confidence"),
    role: str = typer.Option("worker", "--role"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = complete_mission_feature(
            store=store,
            mission_id=mission_id,
            feature_id=feature_id,
            completed_work=completed_work,
            remaining_work=remaining_work,
            known_issues=tuple(known_issue),
            next_recommendation=next_recommendation,
            confidence=confidence,
            role=role,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter("unknown mission or feature for completion request") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        _emit_json(result.to_dict())
        return
    color = "green" if result.status in {"recorded", "completed"} else "white"
    console.print(f"[{color}]{result.status}[/{color}] {result.message}")
    console.print(f"feature_id={result.feature_id}")
    if result.milestone_id:
        console.print(f"milestone_id={result.milestone_id}")
    if result.run_id:
        console.print(f"run_id={result.run_id}")
    if result.handoff_id:
        console.print(f"handoff_id={result.handoff_id}")


@mission_app.command("list-runs")
def mission_list_runs_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    runs = store.list_runs(mission_id=mission_id)
    if not runs:
        console.print("[dim]No mission runs found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Feature", no_wrap=True)
    table.add_column("Milestone", no_wrap=True)
    for item in runs:
        table.add_row(
            item.id,
            item.role if not item.role_model else f"{item.role} ({item.role_model})",
            item.status,
            item.related_feature_id or "—",
            item.related_milestone_id or "—",
        )
    console.print(table)


@mission_app.command("show-run")
def mission_show_run_command(
    run_id: str = typer.Argument(...),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        run = store.load_run(run_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission run: {run_id!r}") from exc
    if json_output:
        _emit_json(run.to_dict())
        return
    console.print(f"[bold]{run.id}[/bold]")
    console.print(f"mission_id={run.mission_id}")
    console.print(f"role={run.role}")
    if run.role_model:
        console.print(f"role_model={run.role_model}")
    console.print(f"status={run.status}")
    if run.related_feature_id:
        console.print(f"related_feature_id={run.related_feature_id}")
    if run.related_milestone_id:
        console.print(f"related_milestone_id={run.related_milestone_id}")
    if run.summary:
        console.print("\n[bold]Summary[/bold]")
        console.print(run.summary)


@mission_app.command("list-handoffs")
def mission_list_handoffs_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    feature_id: str | None = typer.Option(None, "--feature"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    handoffs = store.list_handoffs(mission_id=mission_id, feature_id=feature_id)
    if not handoffs:
        console.print("[dim]No handoffs found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Feature", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Confidence", no_wrap=True)
    table.add_column("Completed Work", overflow="fold")
    for item in handoffs:
        table.add_row(
            item.id,
            item.feature_id,
            item.role if not item.role_model else f"{item.role} ({item.role_model})",
            f"{item.confidence:.2f}",
            item.completed_work,
        )
    console.print(table)


@mission_app.command("show-handoff")
def mission_show_handoff_command(
    handoff_id: str = typer.Argument(...),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        handoff = store.load_handoff(handoff_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission handoff: {handoff_id!r}") from exc
    if json_output:
        _emit_json(handoff.to_dict())
        return
    console.print(f"[bold]{handoff.id}[/bold]")
    console.print(f"mission_id={handoff.mission_id}")
    console.print(f"feature_id={handoff.feature_id}")
    console.print(f"role={handoff.role}")
    if handoff.role_model:
        console.print(f"role_model={handoff.role_model}")
    console.print(f"confidence={handoff.confidence:.2f}")
    console.print("\n[bold]Completed Work[/bold]")
    console.print(handoff.completed_work)
    if handoff.remaining_work:
        console.print("\n[bold]Remaining Work[/bold]")
        console.print(handoff.remaining_work)
    if handoff.known_issues:
        console.print("\n[bold]Known Issues[/bold]")
        for item in handoff.known_issues:
            console.print(f"- {item}")
    if handoff.next_recommendation:
        console.print("\n[bold]Next Recommendation[/bold]")
        console.print(handoff.next_recommendation)


@mission_app.command("validate-milestone")
def mission_validate_milestone_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    milestone_id: str | None = typer.Option(None, "--milestone"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = validate_mission_milestone(
            store=store,
            mission_id=mission_id,
            milestone_id=milestone_id,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter("unknown mission or milestone for validation request") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        _emit_json(result.to_dict())
        return
    color = {
        "passed": "green",
        "completed": "green",
        "failed": "yellow",
    }.get(result.status, "white")
    console.print(f"[{color}]{result.status}[/{color}] {result.message}")
    console.print(f"milestone_id={result.milestone_id}")
    console.print(f"run_id={result.run_id}")
    if result.scrutiny_run_id:
        console.print(f"scrutiny_run_id={result.scrutiny_run_id}")
    if result.behavior_run_id:
        console.print(f"behavior_run_id={result.behavior_run_id}")
    console.print(f"findings_count={result.findings_count}")
    for feature_id in result.corrective_feature_ids:
        console.print(f"corrective_feature_id={feature_id}")


@mission_app.command("execute-milestone")
def mission_execute_milestone_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    milestone_id: str | None = typer.Option(None, "--milestone"),
    max_steps: int = typer.Option(20, "--max-steps"),
    auto_complete: bool = typer.Option(False, "--auto-complete"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = execute_mission_milestone(
            store=store,
            mission_id=mission_id,
            milestone_id=milestone_id,
            max_steps=max_steps,
            auto_complete=auto_complete,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter("unknown mission or milestone for execution request") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        _emit_json(result.to_dict())
        return
    color = {
        "completed": "green",
        "paused": "yellow",
        "blocked": "yellow",
        "no_work": "dim",
    }.get(result.status, "white")
    console.print(
        f"[{color}]{result.status}[/{color}] "
        f"steps={result.steps_run} stop_reason={result.stop_reason}"
    )
    console.print(f"milestone_id={result.milestone_id}")
    _print_mission_steps(result.steps)


@mission_app.command("execute-burst")
def mission_execute_burst_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    max_steps: int = typer.Option(50, "--max-steps"),
    auto_complete: bool = typer.Option(False, "--auto-complete"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = execute_mission_burst(
            store=store,
            mission_id=mission_id,
            max_steps=max_steps,
            auto_complete=auto_complete,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        _emit_json(result.to_dict())
        return
    color = {
        "completed": "green",
        "paused": "yellow",
        "blocked": "yellow",
        "no_work": "dim",
    }.get(result.status, "white")
    console.print(
        f"[{color}]{result.status}[/{color}] "
        f"steps={result.steps_run} stop_reason={result.stop_reason}"
    )
    _print_mission_steps(result.steps)


@mission_app.command("schedule-once")
def mission_schedule_once_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    auto_complete: bool | None = typer.Option(None, "--auto-complete/--no-auto-complete"),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    cfg = load_config(config_path)
    scheduler = cfg.mission_scheduler
    resolved_max_steps = max_steps if max_steps is not None else (scheduler.max_steps or 20)
    resolved_auto_complete = (
        auto_complete if auto_complete is not None else bool(scheduler.auto_complete or False)
    )
    if resolved_max_steps < 1:
        raise typer.BadParameter("--max-steps must be at least 1")
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        result = execute_mission_burst(
            store=store,
            mission_id=mission_id,
            max_steps=resolved_max_steps,
            auto_complete=resolved_auto_complete,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    record_dir = write_mission_scheduled_run_record(
        store=store,
        cwd=working_dir,
        result=result,
    )
    payload = {
        "result": result.to_dict(),
        "record_dir": str(record_dir),
    }
    if json_output:
        _emit_json(payload)
        return
    color = {
        "completed": "green",
        "paused": "yellow",
        "blocked": "yellow",
        "no_work": "dim",
    }.get(result.status, "white")
    console.print(
        f"[{color}]{result.status}[/{color}] "
        f"steps={result.steps_run} stop_reason={result.stop_reason}"
    )
    console.print(f"record_dir={record_dir}")


@mission_app.command("list-findings")
def mission_list_findings_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    milestone_id: str | None = typer.Option(None, "--milestone"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    findings = store.list_findings(mission_id=mission_id, milestone_id=milestone_id)
    if not findings:
        console.print("[dim]No findings found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Milestone", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for item in findings:
        table.add_row(item.id, item.severity, item.milestone_id, item.summary)
    console.print(table)


@mission_app.command("create-opportunity")
def mission_create_opportunity_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    finding_id: str = typer.Option(..., "--finding"),
    title: str | None = typer.Option(None, "--title"),
    summary: str | None = typer.Option(None, "--summary"),
    theme: str | None = typer.Option(None, "--theme"),
    priority: str = typer.Option("medium", "--priority"),
    created_by: str = typer.Option("mission-validator", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    mission_store = MissionStore(root=default_mission_root(working_dir))
    research_store = ResearchStore(root=default_research_root(working_dir))
    try:
        finding = next(
            item
            for item in mission_store.list_findings(mission_id=mission_id)
            if item.id == finding_id
        )
    except StopIteration as exc:
        raise typer.BadParameter(f"unknown mission finding: {finding_id!r}") from exc
    opportunity = Opportunity(
        id=research_store.new_id("opp", title or finding.summary),
        title=(title or f"Mission follow-up: {finding.summary}").strip(),
        summary=(summary or finding.recommended_fix or finding.summary).strip(),
        mission_id=mission_id,
        theme=(theme or "").strip(),
        priority=priority.strip() or "medium",
        created_by=created_by.strip() or "mission-validator",
    )
    target = research_store.add_opportunity(opportunity)
    if json_output:
        _emit_json({"opportunity": opportunity.to_dict(), "path": str(target)})
        return
    console.print(f"[green]Created linked opportunity {opportunity.id}[/green] at {target}")


@mission_app.command("create-candidate")
def mission_create_candidate_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    feature_id: str = typer.Option(..., "--feature"),
    title: str | None = typer.Option(None, "--title"),
    summary: str | None = typer.Option(None, "--summary"),
    expected_metric: str | None = typer.Option(None, "--expected-metric"),
    validation_plan: str | None = typer.Option(None, "--validation-plan"),
    risk_level: str = typer.Option("medium", "--risk-level"),
    created_by: str = typer.Option("mission-reporter", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    mission_store = MissionStore(root=default_mission_root(working_dir))
    research_store = ResearchStore(root=default_research_root(working_dir))
    try:
        feature = mission_store.load_feature(feature_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission feature: {feature_id!r}") from exc
    if feature.mission_id != mission_id:
        raise typer.BadParameter(
            f"feature {feature_id!r} does not belong to mission {mission_id!r}"
        )
    if feature.status not in {"validated", "completed"}:
        raise typer.BadParameter(
            "mission promotion candidate creation requires a completed or validated feature"
        )
    candidate = PromotionCandidate(
        id=research_store.new_id("promo", title or feature.title),
        title=(title or f"Mission candidate: {feature.title}").strip(),
        summary=(summary or feature.summary).strip(),
        mission_id=mission_id,
        mission_feature_ids=(feature_id,),
        target_files=feature.target_files,
        expected_metric=(
            expected_metric or f"feature {feature.title} is accepted by mission validation"
        ).strip(),
        validation_plan=(
            validation_plan or "Run mission validation and the relevant fixture tests."
        ).strip(),
        risk_level=risk_level.strip() or "medium",
        created_by=created_by.strip() or "mission-reporter",
    )
    target = research_store.add_promotion_candidate(candidate)
    if json_output:
        _emit_json({"candidate": candidate.to_dict(), "path": str(target)})
        return
    console.print(f"[green]Created linked promotion candidate {candidate.id}[/green] at {target}")


@mission_app.command("summarize")
def mission_summarize_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        report = build_mission_summary_report(store=store, mission_id=mission_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission: {mission_id!r}") from exc
    target = write_mission_summary_report(store=store, report=report)
    if json_output:
        payload = {
            "report": report.to_dict(),
            "report_dir": str(target),
        }
        _emit_json(payload)
        return
    console.print(f"[green]Wrote mission summary {report.id}[/green] at {target}")
    _print_mission_report(report)


@mission_app.command("list-reports")
def mission_list_reports_command(
    *,
    mission_id: str | None = typer.Option(None, "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    reports = list_mission_reports(store=store, mission_id=mission_id)
    if not reports:
        console.print("[dim]No mission reports found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Mission", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for item in reports:
        table.add_row(item.id, item.mission_id, item.status, item.summary)
    console.print(table)


@mission_app.command("show-report")
def mission_show_report_command(
    report_id: str = typer.Argument(...),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = MissionStore(root=default_mission_root(working_dir))
    try:
        report = load_mission_report(store=store, report_id=report_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown mission report: {report_id!r}") from exc
    if json_output:
        _emit_json(report.to_dict())
        return
    _print_mission_report(report)


__all__ = ["mission_app"]
