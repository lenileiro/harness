from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from harness.cli.config import HarnessConfig
from harness.core import (
    ConsequencePredictor,
    ContextBudget,
    ContextCompactor,
    LLMPlanner,
    Planner,
    RepairOrchestrator,
    RunRequest,
    Verification,
    configure_logging,
)
from harness.storage.sqlite import SQLiteStorage


def run_command(
    *,
    prompt: str,
    model: str | None,
    provider: str | None,
    failover: str | None,
    base_url: str | None,
    cwd: Path | None,
    max_steps: int,
    max_output_tokens: int | None,
    session_id: str | None,
    task_ref: str | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    inbox: bool,
    verify: str | None,
    verify_command: str | None,
    critic: str | None,
    require_tools: bool,
    goal: bool,
    max_context_tokens: int | None,
    predict: bool,
    auto_compact: bool,
    max_repair: int,
    profile: str,
    bare: bool,
    phases: str | None,
    loop_detect: bool,
    contracts: bool,
    tips: bool,
    verbose: bool,
    config_path: Path | None,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    run_async: Any,
    run_once: Any,
) -> None:
    """Run a single prompt through the agent and stream the result to stdout."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    if not yes and os.environ.get("HARNESS_YES"):
        yes = True

    if bare:
        profile = "bare"
    if profile not in ("bare", "minimal", "diagnostic", "strict", "adaptive"):
        console.print(
            "[red]Invalid --profile "
            f"{profile!r}; expected bare, adaptive, minimal, diagnostic, or strict.[/red]"
        )
        raise typer.Exit(2)

    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        run_async(
            run_once(
                prompt=prompt,
                model=effective_model,
                chain=chain,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
                max_output_tokens=max_output_tokens,
                session_id=session_id,
                task_ref=task_ref,
                db=db,
                in_memory=in_memory,
                yes=yes,
                inbox=inbox,
                verify=verify,
                verify_command=verify_command,
                critic=critic,
                require_tools=require_tools,
                goal=goal,
                max_context_tokens=max_context_tokens,
                predict=predict,
                auto_compact=auto_compact,
                max_repair=max_repair,
                profile=profile,
                phases=phases,
                loop_detect=loop_detect,
                contracts=contracts,
                tips=tips,
                config=cfg,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


async def run_once(
    *,
    prompt: str,
    model: str,
    chain: list[str],
    base_url: str | None,
    cwd: Path,
    max_steps: int,
    max_output_tokens: int | None,
    session_id: str | None,
    task_ref: str | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    inbox: bool,
    verify: str | None,
    verify_command: str | None = None,
    critic: str | None = None,
    require_tools: bool = False,
    goal: bool = False,
    max_context_tokens: int | None = None,
    predict: bool = False,
    auto_compact: bool = False,
    max_repair: int = 3,
    profile: str = "minimal",
    phases: str | None = None,
    loop_detect: bool = True,
    contracts: bool = True,
    tips: bool = True,
    config: HarnessConfig,
    build_storage: Any,
    resolve_task_attachment: Any,
    resolve_runtime_strategy: Any,
    build_verifier: Any,
    build_critic: Any,
    build_adapter: Any,
    build_agent: Any,
    print_defense_ledger: Any,
    render: Any,
    default_system_prompt: str,
    console: Console,
) -> None:
    storage = build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        task_id, _task = await resolve_task_attachment(storage, task_ref, session_id)

        strategy = resolve_runtime_strategy(
            prompt=prompt,
            requested_profile=profile,
            verify_command=verify_command,
            phases=phases,
            requested_critic=critic,
        )

        verifier = build_verifier(
            verify,
            chain=chain,
            model=model,
            config=config,
            build_adapter=build_adapter,
            cwd=cwd,
            verify_command=verify_command,
        )
        critic_obj = (
            None
            if strategy.structural_profile == "bare"
            else build_critic(
                strategy.critic_mode,
                chain=chain,
                model=model,
                config=config,
                build_adapter=build_adapter,
            )
        )
        budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        planner: Planner | None = None
        if goal:
            adapter = build_adapter(chain[0], base_url=base_url, config=config)
            planner = LLMPlanner(adapter=adapter, model=model)
        compactor: ContextCompactor | None = None
        if auto_compact:
            adapter = build_adapter(chain[0], base_url=base_url, config=config)
            compactor = ContextCompactor(adapter=adapter, model=model)

        from harness.core import DEFAULT_RESUME_PATH as _DEFAULT_RESUME_PATH
        from harness.core import ArtifactTipProvider as _ArtifactTipProvider
        from harness.core import CompositeTipsProvider as _CompositeTipsProvider
        from harness.core import ContractRegistry as _ContractRegistry
        from harness.core import LoopDetector as _LoopDetector
        from harness.core import ResumeContract as _ResumeContract
        from harness.core import StaticTipsProvider as _StaticTipsProvider
        from harness.core import TipLibrary as _TipLibrary

        loop_detector_obj = (
            _LoopDetector() if (loop_detect and strategy.structural_profile != "bare") else None
        )
        contracts_obj = None
        if contracts and strategy.structural_profile != "bare":
            registry = _ContractRegistry.from_paths(
                [
                    cwd / ".harness" / "contracts",
                    Path.home() / ".harness" / "contracts",
                ]
            )
            if registry:
                contracts_obj = registry
        tips_obj = None
        if tips and strategy.structural_profile != "bare":
            providers: list[object] = []
            library = _TipLibrary.load(
                [
                    cwd / ".harness" / "tips.jsonl",
                    Path.home() / ".harness" / "tips.jsonl",
                ]
            )
            if library:
                providers.append(library)
            experience_paths: list[Path] = []
            configured_roots = os.environ.get("HARNESS_EXPERIENCE_ROOTS", "")
            for raw in configured_roots.split(os.pathsep):
                raw = raw.strip()
                if raw:
                    experience_paths.append(Path(raw))
            repo_runs = cwd / "evals" / "runs"
            if repo_runs not in experience_paths:
                experience_paths.append(repo_runs)
            artifact_provider = _ArtifactTipProvider.load(experience_paths)
            if artifact_provider:
                providers.append(artifact_provider)
            if len(providers) == 1:
                tips_obj = providers[0]
            elif providers:
                tips_obj = _CompositeTipsProvider(providers=providers)  # type: ignore[arg-type]
            else:
                tips_obj = _StaticTipsProvider(tips=[])

        resume_obj = _ResumeContract.load(cwd / _DEFAULT_RESUME_PATH)

        agent = build_agent(
            chain=chain,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
            inbox=inbox,
            activity_store=storage,  # type: ignore[arg-type]
            approval_store=storage,  # type: ignore[arg-type]
            verifier=verifier,
            critic=critic_obj,
            budget=budget,
            memory_store=storage,  # type: ignore[arg-type]
            planner=planner,
            predictor=ConsequencePredictor() if predict else None,
            repair=RepairOrchestrator() if predict else None,
            system_prompt=default_system_prompt,
            compactor=compactor,
            max_repair_attempts=max_repair,
            profile=strategy.structural_profile,
            phases_enabled=bool(phases),
            loop_detector=loop_detector_obj,
            contracts=contracts_obj,
            tips_provider=tips_obj,
            resume=resume_obj,
        )
        if profile == "adaptive":
            console.print(f"[dim]adaptive strategy[/dim] {strategy.rationale}")

        request_kwargs: dict[str, object] = {
            "prompt": prompt,
            "model": model,
            "max_steps": max_steps,
            "require_tool_use": require_tools,
        }
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens
        if session_id:
            request_kwargs["session_id"] = session_id
        if task_id:
            request_kwargs["task_id"] = task_id
        if phases:
            parsed_phases = [p.strip().lower() for p in phases.split(",") if p.strip()]
            if parsed_phases:
                request_kwargs["phases"] = parsed_phases
        request = RunRequest(**request_kwargs)  # type: ignore[arg-type]

        last_verification: Verification | None = None
        try:
            async for event in agent.run(request):
                render(event)
                if isinstance(event, Verification):
                    last_verification = event
        except Exception as exc:
            console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
            raise typer.Exit(1) from None
        await print_defense_ledger(storage, session_id, console=console)
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()

    if last_verification is not None and not last_verification.result.can_finish:
        raise typer.Exit(2)
