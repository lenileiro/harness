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
    Done,
    LLMPlanner,
    Planner,
    RepairOrchestrator,
    RunRequest,
    TextDelta,
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
    domain: str,
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
                domain=domain,
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
    domain: str = "coding",
    phases: str | None = None,
    loop_detect: bool = True,
    contracts: bool = True,
    tips: bool = True,
    silent: bool = False,
    config: HarnessConfig,
    build_storage: Any,
    resolve_task_attachment: Any,
    resolve_runtime_strategy: Any,
    build_verifier: Any,
    build_critic: Any,
    build_adapter: Any,
    build_tools: Any,
    build_agent: Any,
    print_defense_ledger: Any,
    render: Any,
    default_system_prompt: str,
    console: Console,
) -> str | None:
    storage = build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        task_id, _task = await resolve_task_attachment(storage, task_ref, session_id)
        from harness.cli.plugins import (
            load_cli_domain_profile_providers as _load_cli_domain_profile_providers,
        )
        from harness.cli.plugins import (
            load_cli_experience_providers as _load_cli_experience_providers,
        )
        from harness.core import get_domain_profile as _get_domain_profile

        domain_profile = _get_domain_profile(
            domain,
            providers=_load_cli_domain_profile_providers(cwd, config=config),
        )

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
        from harness.core import ContractRegistry as _ContractRegistry
        from harness.core import LoopDetector as _LoopDetector
        from harness.core import ResumeContract as _ResumeContract
        from harness.core import (
            load_default_experience_provider as _load_default_experience_provider,
        )

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
            extra_experience = _load_cli_experience_providers(cwd, config=config)
            tips_obj = _load_default_experience_provider(cwd=cwd)
            if extra_experience:
                tips_obj = _load_default_experience_provider(
                    cwd=cwd,
                    extra_providers=extra_experience,
                )

        resume_obj = _ResumeContract.load(cwd / _DEFAULT_RESUME_PATH)

        allowed_tools = set(domain_profile.allowed_tools) if domain_profile.allowed_tools else None

        # Use a domain-specific tool subset without forking the runtime path.
        def scoped_build_tools(tool_cwd: Path) -> Any:
            return build_tools(
                tool_cwd,
                config=config,
                include=allowed_tools,
            )

        agent = build_agent(
            chain=chain,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
            build_tools=scoped_build_tools,
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
            system_prompt=domain_profile.system_prompt or default_system_prompt,
            compactor=compactor,
            max_repair_attempts=max_repair,
            profile=strategy.structural_profile,
            phases_enabled=bool(phases),
            loop_detector=loop_detector_obj,
            contracts=contracts_obj,
            tips_provider=tips_obj,
            resume=resume_obj,
        )
        if profile == "adaptive" and not silent:
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
        final_text: str | None = None
        streamed_parts: list[str] = []
        try:
            async for event in agent.run(request):
                if not silent:
                    render(event)
                if isinstance(event, TextDelta):
                    streamed_parts.append(event.text)
                elif (
                    isinstance(event, Done)
                    and event.final_message is not None
                    and isinstance(event.final_message.content, str)
                ):
                    final_text = event.final_message.content
                if isinstance(event, Verification):
                    last_verification = event
        except Exception as exc:
            if not silent:
                console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
            raise typer.Exit(1) from None
        if not silent:
            await print_defense_ledger(storage, session_id, console=console)
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()

    if last_verification is not None and not last_verification.result.can_finish:
        raise typer.Exit(2)
    return final_text or ("".join(streamed_parts).strip() or None)
