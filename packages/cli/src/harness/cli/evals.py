from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from harness.cli.common import _build_adapter, _load_cli_config, _truncate, console
from harness.core import correlate_defenses, parse_ledger_text

eval_app = typer.Typer(
    name="eval",
    help="Behavioral eval harness: run fixtures and score agent output.",
    no_args_is_help=True,
)


def _load_eval_module(name: str, evals_root: Path):
    """Load runner.py or judge.py from the evals/ directory at runtime."""
    import importlib.util
    import sys as _sys

    module_name = f"evals.{name}"
    spec = importlib.util.spec_from_file_location(module_name, evals_root / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name}.py from {evals_root}")

    mod = importlib.util.module_from_spec(spec)
    repo_root = str(evals_root.parent)
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    _sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _find_evals_root() -> Path | None:
    """Walk CWD upward looking for evals/fixtures/."""
    current = Path.cwd().resolve()
    while True:
        if (current / "evals" / "fixtures").is_dir():
            return current / "evals"
        parent = current.parent
        if parent == current:
            return None
        current = parent


@eval_app.command("list")
def eval_list(
    fixture_set: Annotated[
        str,
        typer.Option(
            "--fixture-set",
            help="Fixture directory under evals/ to inspect (fixtures, fixtures-mutated, fixtures-holdout).",
        ),
    ] = "fixtures",
    include_holdout: Annotated[
        bool,
        typer.Option("--include-holdout", help="Include fixtures marked holdout in metadata."),
    ] = False,
) -> None:
    """List all available eval fixtures."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    runner = _load_eval_module("runner", evals_root)
    fixtures = runner.discover_fixtures(
        evals_root,
        fixtures_subdir=fixture_set,
        include_holdout=include_holdout,
    )
    if not fixtures:
        console.print("[dim]No fixtures found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Primary Dimension")
    table.add_column("Trap (summary)")
    for fx in fixtures:
        primary = fx.rules.primary_dimension
        trap = fx.rules.trap or ""
        table.add_row(fx.name, fx.family, primary, _truncate(trap.strip(), 70))
    console.print(table)


@eval_app.command("mutate")
def eval_mutate(
    fixture_name: Annotated[
        str,
        typer.Argument(help="Fixture directory under evals/fixtures/ to mutate."),
    ],
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Deterministic seed driving rename choices. Same seed = same mutation.",
        ),
    ] = 1,
    dest: Annotated[
        Path | None,
        typer.Option(
            "--dest",
            help="Override destination root (default: evals/fixtures-mutated/).",
        ),
    ] = None,
) -> None:
    """Apply structure-preserving mutations to one fixture."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    mutator = _load_eval_module("mutator", evals_root)
    src_dir = evals_root / "fixtures" / fixture_name
    if not src_dir.is_dir():
        console.print(f"[red]Fixture {fixture_name!r} not found at {src_dir}.[/red]")
        raise typer.Exit(1)
    result = mutator.mutate_fixture(src_dir, seed=seed, dest_root=dest)
    if not result.renames:
        console.print(
            f"[yellow]No renames applied — the fixture didn't contain any "
            f"of the pool symbols. Copy written to {result.dest_dir}.[/yellow]"
        )
        return
    console.print(f"[green]Mutated {fixture_name} → {result.dest_dir}[/green]")
    console.print("[bold]Renames:[/bold]")
    for original, target in result.renames.items():
        console.print(f"  {original} → {target}")
    if result.touched_files:
        console.print(f"[dim]Touched {len(result.touched_files)} file(s).[/dim]")


@eval_app.command("calibrate")
def eval_calibrate(
    report_path: Annotated[
        Path,
        typer.Argument(help="Path to a saved eval report.json artifact."),
    ],
    gold_dir: Annotated[
        Path | None,
        typer.Option(
            "--gold-dir",
            help="Directory of human-labeled gold trajectory JSON files (default: evals/gold).",
        ),
    ] = None,
) -> None:
    """Compare a saved eval report against gold-labeled trajectory scores."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    calibration = _load_eval_module("calibration", evals_root)
    resolved_gold_dir = gold_dir or (evals_root / "gold")
    labels = calibration.load_gold_labels(resolved_gold_dir)
    if not labels:
        console.print(f"[yellow]No gold labels found in {resolved_gold_dir}.[/yellow]")
        raise typer.Exit(1)
    rows = calibration.compare_report_to_gold(
        calibration.load_report(report_path),
        labels,
    )
    if not rows:
        console.print("[yellow]No overlapping labeled runs were found in the report.[/yellow]")
        raise typer.Exit(1)
    table = Table(show_header=True, header_style="bold", title="Judge calibration")
    table.add_column("Dimension", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("Exact", justify="right")
    table.add_column("MAE", justify="right")
    for row in rows:
        table.add_row(
            row.dimension,
            str(row.count),
            f"{row.exact_match_rate:.2f}",
            f"{row.mean_absolute_error:.2f}",
        )
    console.print(table)


@eval_app.command("history")
def eval_history(
    limit: Annotated[
        int,
        typer.Option("--limit", help="Number of history rows to display."),
    ] = 10,
) -> None:
    """Show recent saved eval runs from evals/results/history.jsonl."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    history_path = evals_root / "results" / "history.jsonl"
    if not history_path.exists():
        console.print("[dim]No eval history found.[/dim]")
        return
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = rows[-limit:]
    table = Table(show_header=True, header_style="bold", title="Eval history")
    table.add_column("Run", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Judge", no_wrap=True)
    table.add_column("Set", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("Results", justify="right")
    for row in rows:
        table.add_row(
            row.get("run_id", "?"),
            f"{row.get('provider', '?')}/{row.get('model', '?')}",
            f"{row.get('judge_provider', '?')}/{row.get('judge_model', '?')}",
            row.get("fixture_set", "?"),
            row.get("benchmark_mode", "original"),
            str(row.get("n_runs", "?")),
            str(len(row.get("results", []))),
        )
    console.print(table)


def _collect_adjustment_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name == "harness_adjustments.json" else []
    if not root.exists():
        return []
    return sorted(root.rglob("harness_adjustments.json"))


def _load_adjustments(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _collect_adjustment_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("source_file", str(path))
            rows.append(row)
    return rows


@eval_app.command("adjustments")
def eval_adjustments(
    root: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Run directory, artifact directory, or harness_adjustments.json file. "
                "Defaults to evals/runs."
            )
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of adjustment rows to display."),
    ] = 20,
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Filter by adjustment kind."),
    ] = None,
) -> None:
    """Inspect analyzed harness adjustments from saved eval artifacts."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    search_root = root or (evals_root / "runs")
    rows = _load_adjustments(search_root)
    if kind:
        rows = [row for row in rows if str(row.get("kind", "")).strip() == kind]
    if not rows:
        console.print("[dim]No analyzed harness adjustments found.[/dim]")
        return
    rows = rows[:limit]
    table = Table(show_header=True, header_style="bold", title="Harness adjustments")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Variant", no_wrap=True)
    table.add_column("Weight", justify="right")
    table.add_column("Text")
    for row in rows:
        raw_weight = row.get("weight", 0.0)
        if isinstance(raw_weight, int | float | str):
            try:
                weight_text = f"{float(raw_weight):.1f}"
            except ValueError:
                weight_text = "0.0"
        else:
            weight_text = "0.0"
        table.add_row(
            str(row.get("kind", "?")),
            str(row.get("source_fixture_name", "?")),
            str(row.get("source_variant", "?")),
            weight_text,
            str(row.get("text", "")),
        )
    console.print(table)


@eval_app.command("export-adjustments")
def eval_export_adjustments(
    output: Annotated[
        Path,
        typer.Argument(help="Destination file (.json or .jsonl)."),
    ],
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            help="Run directory, artifact directory, or harness_adjustments.json file. Defaults to evals/runs.",
        ),
    ] = None,
) -> None:
    """Export a consolidated adjustment corpus from saved eval artifacts."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    search_root = root or (evals_root / "runs")
    rows = _load_adjustments(search_root)
    if not rows:
        console.print("[dim]No analyzed harness adjustments found.[/dim]")
        raise typer.Exit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    else:
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    console.print(f"[green]Exported {len(rows)} adjustment(s) to {output}[/green]")


@eval_app.command("validate")
def eval_validate() -> None:
    """Validate eval assets: fixture sets, suites, and gold-label coverage."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)
    runner = _load_eval_module("runner", evals_root)
    review_runner = _load_eval_module("review_runner", evals_root)
    research_runner = _load_eval_module("research_runner", evals_root)
    docs_runner = _load_eval_module("docs_runner", evals_root)
    workflow_runner = _load_eval_module("workflow_runner", evals_root)
    calibration = _load_eval_module("calibration", evals_root)

    fixture_sets = ("fixtures", "fixtures-mutated", "fixtures-holdout")
    suite_map = {
        "fixtures": "full",
        "fixtures-mutated": "mutated",
        "fixtures-holdout": "holdout",
    }
    discovered: dict[str, list[Any]] = {}
    for fixture_set in fixture_sets:
        discovered[fixture_set] = runner.discover_fixtures(
            evals_root,
            fixtures_subdir=fixture_set,
            include_holdout=True,
        )
        if not discovered[fixture_set]:
            console.print(f"[red]No fixtures discovered in {fixture_set}.[/red]")
            raise typer.Exit(1)

    suites_dir = evals_root / "suites"
    for fixture_set, suite_name in suite_map.items():
        suite_path = suites_dir / f"{suite_name}.txt"
        if not suite_path.exists():
            console.print(f"[red]Missing suite file:[/red] {suite_path}")
            raise typer.Exit(1)
        members = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixture_names = {fixture.name for fixture in discovered[fixture_set]}
        missing = sorted(members - fixture_names)
        if missing:
            console.print(
                f"[red]Suite {suite_name} references unknown fixtures:[/red] {', '.join(missing)}"
            )
            raise typer.Exit(1)

    review_fixtures = review_runner.discover_review_fixtures(evals_root)
    if not review_fixtures:
        console.print("[red]No review fixtures discovered in review-fixtures.[/red]")
        raise typer.Exit(1)
    review_suite_path = suites_dir / "review-smoke.txt"
    if not review_suite_path.exists():
        console.print(f"[red]Missing suite file:[/red] {review_suite_path}")
        raise typer.Exit(1)
    review_members = {
        line.strip()
        for line in review_suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    review_names = {fixture.name for fixture in review_fixtures}
    missing_review = sorted(review_members - review_names)
    if missing_review:
        console.print(
            "[red]Review suite references unknown fixtures:[/red] " + ", ".join(missing_review)
        )
        raise typer.Exit(1)

    research_fixtures = research_runner.discover_research_fixtures(evals_root)
    if not research_fixtures:
        console.print("[red]No research fixtures discovered in research-fixtures.[/red]")
        raise typer.Exit(1)
    research_suite_path = suites_dir / "research-smoke.txt"
    if not research_suite_path.exists():
        console.print(f"[red]Missing suite file:[/red] {research_suite_path}")
        raise typer.Exit(1)
    research_members = {
        line.strip()
        for line in research_suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    research_names = {fixture.name for fixture in research_fixtures}
    missing_research = sorted(research_members - research_names)
    if missing_research:
        console.print(
            "[red]Research suite references unknown fixtures:[/red] " + ", ".join(missing_research)
        )
        raise typer.Exit(1)

    docs_fixtures = docs_runner.discover_docs_fixtures(evals_root)
    if not docs_fixtures:
        console.print("[red]No docs fixtures discovered in docs-fixtures.[/red]")
        raise typer.Exit(1)
    docs_suite_path = suites_dir / "docs-smoke.txt"
    if not docs_suite_path.exists():
        console.print(f"[red]Missing suite file:[/red] {docs_suite_path}")
        raise typer.Exit(1)
    docs_members = {
        line.strip()
        for line in docs_suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    docs_names = {fixture.name for fixture in docs_fixtures}
    missing_docs = sorted(docs_members - docs_names)
    if missing_docs:
        console.print(
            "[red]Docs suite references unknown fixtures:[/red] " + ", ".join(missing_docs)
        )
        raise typer.Exit(1)

    workflow_fixtures = workflow_runner.discover_workflow_fixtures(evals_root)
    if not workflow_fixtures:
        console.print("[red]No workflow fixtures discovered in workflow-fixtures.[/red]")
        raise typer.Exit(1)
    workflow_suite_path = suites_dir / "workflow-smoke.txt"
    if not workflow_suite_path.exists():
        console.print(f"[red]Missing suite file:[/red] {workflow_suite_path}")
        raise typer.Exit(1)
    workflow_members = {
        line.strip()
        for line in workflow_suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    workflow_names = {fixture.name for fixture in workflow_fixtures}
    missing_workflow = sorted(workflow_members - workflow_names)
    if missing_workflow:
        console.print(
            "[red]Workflow suite references unknown fixtures:[/red] " + ", ".join(missing_workflow)
        )
        raise typer.Exit(1)

    labels = calibration.load_gold_labels(evals_root / "gold")
    if not labels:
        console.print("[red]No gold labels found.[/red]")
        raise typer.Exit(1)
    gold_keys = {(label.fixture_name, label.variant) for label in labels}
    missing_gold: list[str] = []
    for fixtures in discovered.values():
        for fixture in fixtures:
            for variant in ("defended", "bare"):
                key = (fixture.name, variant)
                if key not in gold_keys:
                    missing_gold.append(f"{fixture.name}:{variant}")
    if missing_gold:
        console.print(
            "[red]Missing gold labels for fixtures:[/red] " + ", ".join(sorted(missing_gold)[:10])
        )
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold", title="Eval asset validation")
    table.add_column("Fixture set", no_wrap=True)
    table.add_column("Count", justify="right")
    for fixture_set in fixture_sets:
        table.add_row(fixture_set, str(len(discovered[fixture_set])))
    table.add_row("review-fixtures", str(len(review_fixtures)))
    table.add_row("research-fixtures", str(len(research_fixtures)))
    table.add_row("docs-fixtures", str(len(docs_fixtures)))
    table.add_row("workflow-fixtures", str(len(workflow_fixtures)))
    table.add_row("gold labels", str(len(labels)))
    console.print(table)


@eval_app.command("review")
def eval_review(
    fixture_name: Annotated[
        str | None,
        typer.Argument(help="Review fixture to run (e.g. 01-missing-none-guard). Omit to run all."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider for the review agent."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for the review agent."),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite",
            help="Optional review suite file under evals/suites/ (for example: review-smoke).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the run artifact directory (default: evals/runs/<timestamp>-review).",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Review command timeout per fixture in seconds."),
    ] = 180,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", help="Cap review output tokens."),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json-out", help="Print the full machine-readable review eval report."),
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run code-review eval fixtures against `harness review`."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    cfg = _load_cli_config(config_path)
    resolved_provider = provider or cfg.default_provider or "ollama"
    resolved_model = model or cfg.default_model or "llama3.2"
    review_runner = _load_eval_module("review_runner", evals_root)

    fixtures = review_runner.discover_review_fixtures(evals_root)
    if suite:
        suite_path = evals_root / "suites" / f"{suite}.txt"
        if not suite_path.exists():
            console.print(f"[red]Suite not found:[/red] {suite_path}")
            raise typer.Exit(1)
        wanted = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixtures = [fixture for fixture in fixtures if fixture.name in wanted]
    if fixture_name:
        fixtures = [fixture for fixture in fixtures if fixture.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Review fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)
    if not fixtures:
        console.print("[dim]No review fixtures to run.[/dim]")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-review"
    artifact_root = output_dir or (evals_root / "runs" / run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    for fixture in fixtures:
        console.print(f"\n[bold blue]▶ review {fixture.name}[/bold blue]")
        result = review_runner.run_review_fixture(
            fixture,
            provider=resolved_provider,
            model=resolved_model,
            artifact_dir=artifact_root / fixture.name,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        results.append(result)
        pass_label = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"  {pass_label} findings={result.findings_count} matched={result.matched_expectations} "
            f"secs={result.duration_seconds:.1f}"
        )
        if result.missing_expectations:
            console.print("  [dim]missing[/dim] " + ", ".join(result.missing_expectations))
        if result.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {result.artifact_dir}")

    report = review_runner.ReviewEvalReport(
        run_id=run_id,
        provider=resolved_provider,
        model=resolved_model,
        artifact_root=artifact_root,
        results=results,
    )
    report_path = artifact_root / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(show_header=True, header_style="bold", title="Review eval")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Pass?", justify="center")
    table.add_column("Findings", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Seconds", justify="right")
    for result in results:
        table.add_row(
            result.fixture_name,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            str(result.findings_count),
            str(result.matched_expectations),
            f"{result.duration_seconds:.1f}",
        )
    console.print(table)
    console.print(f"[dim]report[/dim] {report_path}")


@eval_app.command("research")
def eval_research(
    fixture_name: Annotated[
        str | None,
        typer.Argument(
            help="Research fixture to run (e.g. 01-persistence-tradeoffs). Omit to run all."
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider for the research agent."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for the research agent."),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite",
            help="Optional research suite file under evals/suites/ (for example: research-smoke).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the run artifact directory (default: evals/runs/<timestamp>-research).",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Research command timeout per fixture in seconds."),
    ] = 180,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Research command step budget per fixture."),
    ] = 40,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", help="Cap research output tokens."),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json-out", help="Print the full machine-readable research eval report."),
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run research eval fixtures against `harness research`."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    cfg = _load_cli_config(config_path)
    resolved_provider = provider or cfg.default_provider or "ollama"
    resolved_model = model or cfg.default_model or "llama3.2"
    research_runner = _load_eval_module("research_runner", evals_root)

    fixtures = research_runner.discover_research_fixtures(evals_root)
    if suite:
        suite_path = evals_root / "suites" / f"{suite}.txt"
        if not suite_path.exists():
            console.print(f"[red]Suite not found:[/red] {suite_path}")
            raise typer.Exit(1)
        wanted = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixtures = [fixture for fixture in fixtures if fixture.name in wanted]
    if fixture_name:
        fixtures = [fixture for fixture in fixtures if fixture.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Research fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)
    if not fixtures:
        console.print("[dim]No research fixtures to run.[/dim]")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-research"
    artifact_root = output_dir or (evals_root / "runs" / run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    for fixture in fixtures:
        console.print(f"\n[bold blue]▶ research {fixture.name}[/bold blue]")
        result = research_runner.run_research_fixture(
            fixture,
            provider=resolved_provider,
            model=resolved_model,
            artifact_dir=artifact_root / fixture.name,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            max_steps=max_steps,
        )
        results.append(result)
        pass_label = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"  {pass_label} findings={result.findings_count} sources={result.source_count} "
            f"matched={result.matched_findings}/{result.matched_sources} "
            f"secs={result.duration_seconds:.1f}"
        )
        if result.missing_expectations:
            console.print("  [dim]missing[/dim] " + ", ".join(result.missing_expectations))
        if result.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {result.artifact_dir}")

    report = research_runner.ResearchEvalReport(
        run_id=run_id,
        provider=resolved_provider,
        model=resolved_model,
        artifact_root=artifact_root,
        results=results,
    )
    report_path = artifact_root / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(show_header=True, header_style="bold", title="Research eval")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Pass?", justify="center")
    table.add_column("Findings", justify="right")
    table.add_column("Sources", justify="right")
    table.add_column("Seconds", justify="right")
    for result in results:
        table.add_row(
            result.fixture_name,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            str(result.findings_count),
            str(result.source_count),
            f"{result.duration_seconds:.1f}",
        )
    console.print(table)
    console.print(f"[dim]report[/dim] {report_path}")


@eval_app.command("docs-audit")
def eval_docs_audit(
    fixture_name: Annotated[
        str | None,
        typer.Argument(help="Docs fixture to run (e.g. 01-missing-plugin-docs). Omit to run all."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider for the docs-audit agent."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for the docs-audit agent."),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite",
            help="Optional docs suite file under evals/suites/ (for example: docs-smoke).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the run artifact directory (default: evals/runs/<timestamp>-docs).",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Docs-audit command timeout per fixture in seconds."),
    ] = 180,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", help="Cap docs-audit output tokens."),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json-out", help="Print the full machine-readable docs eval report."),
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run docs-audit eval fixtures against `harness docs-audit`."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    cfg = _load_cli_config(config_path)
    resolved_provider = provider or cfg.default_provider or "ollama"
    resolved_model = model or cfg.default_model or "llama3.2"
    docs_runner = _load_eval_module("docs_runner", evals_root)

    fixtures = docs_runner.discover_docs_fixtures(evals_root)
    if suite:
        suite_path = evals_root / "suites" / f"{suite}.txt"
        if not suite_path.exists():
            console.print(f"[red]Suite not found:[/red] {suite_path}")
            raise typer.Exit(1)
        wanted = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixtures = [fixture for fixture in fixtures if fixture.name in wanted]
    if fixture_name:
        fixtures = [fixture for fixture in fixtures if fixture.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Docs fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)
    if not fixtures:
        console.print("[dim]No docs fixtures to run.[/dim]")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-docs"
    artifact_root = output_dir or (evals_root / "runs" / run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    for fixture in fixtures:
        console.print(f"\n[bold blue]▶ docs {fixture.name}[/bold blue]")
        result = docs_runner.run_docs_fixture(
            fixture,
            provider=resolved_provider,
            model=resolved_model,
            artifact_dir=artifact_root / fixture.name,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        results.append(result)
        pass_label = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"  {pass_label} findings={result.findings_count} matched={result.matched_expectations} "
            f"topics={result.matched_topics} secs={result.duration_seconds:.1f}"
        )
        if result.missing_expectations:
            console.print("  [dim]missing[/dim] " + ", ".join(result.missing_expectations))
        if result.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {result.artifact_dir}")

    report = docs_runner.DocsEvalReport(
        run_id=run_id,
        provider=resolved_provider,
        model=resolved_model,
        artifact_root=artifact_root,
        results=results,
    )
    report_path = artifact_root / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(show_header=True, header_style="bold", title="Docs eval")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Pass?", justify="center")
    table.add_column("Findings", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Topics", justify="right")
    table.add_column("Seconds", justify="right")
    for result in results:
        table.add_row(
            result.fixture_name,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            str(result.findings_count),
            str(result.matched_expectations),
            str(result.matched_topics),
            f"{result.duration_seconds:.1f}",
        )
    console.print(table)
    console.print(f"[dim]report[/dim] {report_path}")


@eval_app.command("workflow")
def eval_workflow(
    fixture_name: Annotated[
        str | None,
        typer.Argument(
            help="Workflow fixture to run (e.g. 01-research-publication-cycle). Omit to run all."
        ),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite",
            help="Optional workflow suite file under evals/suites/ (for example: workflow-smoke).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the run artifact directory (default: evals/runs/<timestamp>-workflow).",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Workflow command timeout per fixture in seconds."),
    ] = 180,
    json_out: Annotated[
        bool,
        typer.Option("--json-out", help="Print the full machine-readable workflow eval report."),
    ] = False,
) -> None:
    """Run deterministic workflow eval fixtures against local CLI feature flows."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    workflow_runner = _load_eval_module("workflow_runner", evals_root)
    fixtures = workflow_runner.discover_workflow_fixtures(evals_root)
    if suite:
        suite_path = evals_root / "suites" / f"{suite}.txt"
        if not suite_path.exists():
            console.print(f"[red]Suite not found:[/red] {suite_path}")
            raise typer.Exit(1)
        wanted = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixtures = [fixture for fixture in fixtures if fixture.name in wanted]
    if fixture_name:
        fixtures = [fixture for fixture in fixtures if fixture.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Workflow fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)
    if not fixtures:
        console.print("[dim]No workflow fixtures to run.[/dim]")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-workflow"
    artifact_root = output_dir or (evals_root / "runs" / run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    for fixture in fixtures:
        console.print(f"\n[bold blue]▶ workflow {fixture.name}[/bold blue]")
        result = workflow_runner.run_workflow_fixture(
            fixture,
            artifact_dir=artifact_root / fixture.name,
            timeout=timeout,
        )
        results.append(result)
        pass_label = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"  {pass_label} steps={result.steps_passed}/{result.steps_total} "
            f"secs={result.duration_seconds:.1f}"
        )
        if result.step_results and not result.passed:
            failed_step = next((step for step in result.step_results if not step.passed), None)
            if failed_step is not None and failed_step.failures:
                console.print("  [dim]failed[/dim] " + ", ".join(failed_step.failures))
        if result.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {result.artifact_dir}")

    report = workflow_runner.WorkflowEvalReport(
        run_id=run_id,
        artifact_root=artifact_root,
        results=results,
    )
    report_path = artifact_root / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(show_header=True, header_style="bold", title="Workflow eval")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Pass?", justify="center")
    table.add_column("Steps", justify="right")
    table.add_column("Seconds", justify="right")
    for result in results:
        table.add_row(
            result.fixture_name,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            f"{result.steps_passed}/{result.steps_total}",
            f"{result.duration_seconds:.1f}",
        )
    console.print(table)
    console.print(f"[dim]report[/dim] {report_path}")


@eval_app.command("run")
def eval_run(
    fixture_name: Annotated[
        str | None,
        typer.Argument(help="Fixture to run (e.g. 01-reproduce-before-repair). Omit to run all."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider for the agent."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for the agent."),
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option("--judge-model", help="Model for the judge (defaults to --model)."),
    ] = None,
    judge_provider: Annotated[
        str | None,
        typer.Option(
            "--judge-provider",
            help="Provider for the judge (defaults to --provider, or ollama when provider=claude).",
        ),
    ] = None,
    no_judge: Annotated[
        bool,
        typer.Option(
            "--no-judge",
            help="Skip LLM judging and report hard pass-rate metrics only.",
        ),
    ] = False,
    agent_timeout: Annotated[
        int,
        typer.Option("--timeout", help="Agent timeout per fixture in seconds."),
    ] = 300,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Cap model output tokens for each eval agent turn.",
        ),
    ] = None,
    n_runs: Annotated[
        int,
        typer.Option(
            "--n-runs",
            help=(
                "Run each fixture N times to measure variance. Reports median "
                "score per dimension and (min..max) range in the final table. "
                "Use 3+ on non-deterministic local models. Default 1."
            ),
        ),
    ] = 1,
    ab: Annotated[
        bool,
        typer.Option(
            "--ab",
            help=(
                "A/B mode: run each fixture twice per rep — once with the "
                "full harness defense chain (defended), once with --bare "
                "(no structural verifiers, no critic). Reports both arms "
                "side-by-side so you can measure the harness's value-add."
            ),
        ),
    ] = False,
    fixture_set: Annotated[
        str,
        typer.Option(
            "--fixture-set",
            help="Fixture directory under evals/ to use (fixtures, fixtures-mutated, fixtures-holdout).",
        ),
    ] = "fixtures",
    benchmark_mode: Annotated[
        str,
        typer.Option(
            "--benchmark-mode",
            help="Fixture selection mode: original, mutated, or mixed.",
        ),
    ] = "original",
    mutation_seeds: Annotated[
        str,
        typer.Option(
            "--mutation-seeds",
            help="Comma-separated deterministic seeds used for mutated/mixed benchmark modes.",
        ),
    ] = "1",
    include_holdout: Annotated[
        bool,
        typer.Option(
            "--include-holdout",
            help="Include fixtures marked holdout in fixture.yaml metadata.",
        ),
    ] = False,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite",
            help="Optional suite file name under evals/suites/ (for example: smoke).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the run artifact directory (default: evals/runs/<timestamp>).",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json-out", help="Print the full machine-readable eval report as JSON."),
    ] = False,
    save_history: Annotated[
        bool,
        typer.Option(
            "--save-history/--no-save-history",
            help="Append a compact summary record to evals/results/history.jsonl.",
        ),
    ] = True,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run one or all eval fixtures and display scored results."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    cfg = _load_cli_config(config_path)
    resolved_provider = provider or cfg.default_provider or "ollama"
    resolved_model = model or cfg.default_model or "llama3.2"
    resolved_judge_model = judge_model or resolved_model
    resolved_judge_provider = judge_provider or (
        "ollama" if resolved_provider == "claude" else resolved_provider
    )

    runner = _load_eval_module("runner", evals_root)
    judge_mod = None if no_judge else _load_eval_module("judge", evals_root)
    mutator = _load_eval_module("mutator", evals_root)
    types_mod = _load_eval_module("types", evals_root)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = output_dir or (evals_root / "runs" / run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    history_path = evals_root / "results" / "history.jsonl"

    seed_values = [int(part.strip()) for part in mutation_seeds.split(",") if part.strip()]
    mutation_coverage: float | None = None
    discovery_root = evals_root
    discovery_subdir = fixture_set
    if benchmark_mode != "original":
        materialized_root = artifact_root / "generated-fixtures"
        fixture_source_root = evals_root / fixture_set
        materialized = mutator.materialize_fixture_set(
            fixture_source_root,
            dest_root=materialized_root,
            mode=benchmark_mode,
            seeds=seed_values,
        )
        mutation_coverage = materialized.mutation_coverage
        discovery_root = materialized_root
        discovery_subdir = "fixtures"

    fixtures = runner.discover_fixtures(
        discovery_root,
        fixtures_subdir=discovery_subdir,
        include_holdout=include_holdout,
    )
    if suite:
        suite_path = evals_root / "suites" / f"{suite}.txt"
        if not suite_path.exists():
            console.print(f"[red]Suite not found:[/red] {suite_path}")
            raise typer.Exit(1)
        wanted = {
            line.strip()
            for line in suite_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        fixtures = [f for f in fixtures if f.name in wanted]
    if fixture_name:
        fixtures = [f for f in fixtures if f.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)

    if not fixtures:
        console.print("[dim]No fixtures to run.[/dim]")
        return

    judge_adapter = (
        None if no_judge else _build_adapter(resolved_judge_provider, base_url=None, config=cfg)
    )
    if benchmark_mode != "original" and mutation_coverage is not None:
        console.print(
            f"[dim]benchmark mode[/dim] {benchmark_mode}  "
            f"[dim]mutation coverage[/dim] {mutation_coverage:.2f}"
        )

    def _score_cell(score: int) -> str:
        color = "green" if score >= 4 else ("yellow" if score == 3 else "red")
        return f"[{color}]{score}/5[/{color}]"

    _DIM_ORDER = (
        ("verification", "Verif."),
        ("scope", "Scope"),
        ("decomposition", "Decomp."),
        ("correctness", "Correct."),
        ("pushback", "Pushback"),
        ("epistemic", "Epist."),
        ("overall", "Overall"),
    )

    def _print_per_fixture(r: Any) -> None:
        row_table = Table(show_header=True, header_style="bold")
        row_table.add_column("Fixture", no_wrap=True)
        for _, col in _DIM_ORDER:
            row_table.add_column(col, justify="center")
        row_table.add_column("Pass?", justify="center")
        row_table.add_row(
            r.fixture_name,
            *(_score_cell(getattr(r, dim).score) for dim, _ in _DIM_ORDER),
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
        )
        console.print(row_table)
        for dim_name, _ in _DIM_ORDER:
            dim = getattr(r, dim_name)
            color = "green" if dim.score >= 4 else ("yellow" if dim.score == 3 else "red")
            console.print(
                f"  [{color}]{dim.score}/5[/{color}] [dim]{dim_name}[/dim]  {dim.rationale}"
            )
        if r.hard_metrics is not None:
            metrics = r.hard_metrics
            console.print(
                "  [dim]hard metrics[/dim] "
                f"pass={metrics.verify_passed} files={metrics.files_touched} "
                f"+{metrics.lines_added}/-{metrics.lines_deleted} "
                f"tools={metrics.tool_calls} verify={metrics.did_run_verification}"
            )
        if r.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {r.artifact_dir}")

    def _print_per_fixture_hard(
        *, fixture_name: str, variant: str, run_index: int, outcome: Any
    ) -> None:
        metrics = outcome.hard_metrics
        if metrics is None:
            console.print(f"  [red]no hard metrics[/red] [{variant}] (run {run_index})")
            return
        label = f"{fixture_name} [dim]({variant}, run {run_index})[/dim]"
        pass_label = "[green]PASS[/green]" if metrics.verify_passed else "[red]FAIL[/red]"
        console.print(
            f"{label} {pass_label}  "
            f"agent_exit={outcome.agent_exit_code} verify_exit={outcome.test_exit_code}  "
            f"files={metrics.files_touched} +{metrics.lines_added}/-{metrics.lines_deleted}  "
            f"tools={metrics.tool_calls} secs={metrics.total_duration_seconds:.1f}"
        )
        if outcome.artifact_dir is not None:
            console.print(f"  [dim]artifacts[/dim] {outcome.artifact_dir}")

    variants: tuple[str, ...] = ("defended", "bare") if ab else ("defended",)

    runs_by_pair: dict[tuple[str, str], list[Any]] = {}
    outcomes_by_pair: dict[tuple[str, str], list[Any]] = {}
    defense_trials: list[tuple[Any, bool, str, str]] = []
    all_results: list[Any] = []
    hard_results: list[dict[str, Any]] = []
    for fx in fixtures:
        console.print(f"\n[bold blue]▶ {fx.name}[/bold blue]")
        agent_desc = (
            resolved_provider
            if resolved_provider == "claude"
            else f"{resolved_provider}/{resolved_model}"
        )
        for variant in variants:
            for run_idx in range(n_runs):
                pieces: list[str] = []
                if ab:
                    pieces.append(f"[{variant}]")
                if n_runs > 1:
                    pieces.append(f"(run {run_idx + 1}/{n_runs})")
                run_label = " " + " ".join(pieces) if pieces else ""
                run_artifact_dir = artifact_root / fx.name / variant / f"run-{run_idx + 1:02d}"
                with console.status(f"[dim]running agent ({agent_desc}){run_label}...[/dim]"):
                    try:
                        outcome = runner.run_fixture(
                            fx,
                            provider=resolved_provider,
                            model=resolved_model,
                            agent_timeout=agent_timeout,
                            max_output_tokens=max_output_tokens,
                            variant=variant,
                            artifact_dir=run_artifact_dir,
                        )
                    except Exception as exc:
                        console.print(f"  [red]run failed{run_label}:[/red] {exc}")
                        continue

                exit_icon = (
                    "[green]✓[/green]" if outcome.agent_exit_code == 0 else "[yellow]![/yellow]"
                )
                test_icon = "[green]✓[/green]" if outcome.test_exit_code == 0 else "[red]✗[/red]"
                console.print(
                    f"  agent {exit_icon} (exit {outcome.agent_exit_code})  "
                    f"tests {test_icon} (exit {outcome.test_exit_code}){run_label}"
                )
                outcomes_by_pair.setdefault((fx.name, variant), []).append(outcome)

                ledger = parse_ledger_text(outcome.transcript)
                if no_judge:
                    passed = bool(outcome.hard_metrics and outcome.hard_metrics.verify_passed)
                    defense_trials.append((ledger, passed, variant, fx.name))
                    hard_results.append(
                        {
                            "fixture_name": fx.name,
                            "variant": variant,
                            "run_index": run_idx + 1,
                            "hard_metrics": (
                                outcome.hard_metrics.to_dict()
                                if outcome.hard_metrics is not None
                                else None
                            ),
                            "artifact_dir": str(run_artifact_dir),
                            "agent_exit_code": outcome.agent_exit_code,
                            "verify_exit_code": outcome.test_exit_code,
                        }
                    )
                    _print_per_fixture_hard(
                        fixture_name=fx.name,
                        variant=variant,
                        run_index=run_idx + 1,
                        outcome=outcome,
                    )
                    continue

                with console.status(f"[dim]scoring{run_label}...[/dim]"):
                    try:
                        assert judge_mod is not None
                        assert judge_adapter is not None
                        result = judge_mod.judge(
                            adapter=judge_adapter,
                            model=resolved_judge_model,
                            fixture_name=fx.name,
                            task_text=fx.task_text,
                            eval_md=fx.eval_md,
                            transcript=outcome.transcript,
                            git_diff=outcome.git_diff,
                            test_output=outcome.test_output,
                            hard_metrics=outcome.hard_metrics,
                            artifact_dir=str(run_artifact_dir),
                            variant=variant,
                            run_index=run_idx + 1,
                        )
                        runs_by_pair.setdefault((fx.name, variant), []).append(result)
                        all_results.append(result)
                    except Exception as exc:
                        console.print(f"  [red]judge failed{run_label}:[/red] {exc}")
                        continue

                defense_trials.append((ledger, result.passed, variant, fx.name))
                _print_per_fixture(result)

    if not runs_by_pair and not outcomes_by_pair:
        return

    def _cell_for_runs(runs: list[Any], dim_name: str) -> str:
        scores = sorted(getattr(r, dim_name).score for r in runs)
        median = statistics.median(scores)
        median_round = round(median)
        color = "green" if median_round >= 4 else ("yellow" if median_round == 3 else "red")
        median_str = f"{median:g}" if median != int(median) else str(int(median))
        if len(scores) == 1:
            return f"[{color}]{median_str}/5[/{color}]"
        return f"[{color}]{median_str}/5[/{color}] [dim]({scores[0]}..{scores[-1]})[/dim]"

    def _aggregate_dimensions(runs: list[Any]) -> dict[str, Any]:
        dims: dict[str, Any] = {}
        for dim_name, _ in _DIM_ORDER:
            scores = [getattr(r, dim_name).score for r in runs]
            sem = None
            if len(scores) >= 2:
                sem = statistics.stdev(scores) / (len(scores) ** 0.5)
            dims[dim_name] = types_mod.AggregatedDimension(
                median=statistics.median(scores),
                minimum=min(scores),
                maximum=max(scores),
                mean=statistics.fmean(scores),
                sem=sem,
            )
        return dims

    def _aggregate_hard_metrics(runs: list[Any]) -> dict[str, float]:
        aggregates: dict[str, float] = {}
        if not runs or runs[0].hard_metrics is None:
            return aggregates
        keys = (
            "files_touched",
            "lines_added",
            "lines_deleted",
            "tool_calls",
            "shell_commands",
            "agent_duration_seconds",
            "verify_duration_seconds",
            "total_duration_seconds",
            "redundant_tool_calls",
            "retry_loops",
        )
        for key in keys:
            values = [
                float(getattr(r.hard_metrics, key)) for r in runs if r.hard_metrics is not None
            ]
            if values:
                aggregates[f"avg_{key}"] = statistics.fmean(values)
        aggregates["verify_pass_rate"] = statistics.fmean(
            [1.0 if r.hard_metrics and r.hard_metrics.verify_passed else 0.0 for r in runs]
        )
        return aggregates

    aggregates: list[Any] = []

    console.print()
    title_pieces = ["Eval Results"]
    if no_judge:
        title_pieces.append("hard metrics only")
    if n_runs > 1:
        title_pieces.append(f"{n_runs} runs each")
    if ab:
        title_pieces.append("A/B: defended vs bare")
    title = " — ".join(title_pieces)
    table = Table(show_header=True, header_style="bold", title=title)
    table.add_column("Fixture / variant", no_wrap=True)
    if not no_judge:
        for _, col in _DIM_ORDER:
            table.add_column(col, justify="center")
    table.add_column("Pass rate", justify="center")
    aggregate_source = outcomes_by_pair if no_judge else runs_by_pair
    for (fx_name, variant), runs in aggregate_source.items():
        if no_judge:
            n_passed = sum(
                1 for outcome in runs if outcome.hard_metrics and outcome.hard_metrics.verify_passed
            )
        else:
            n_passed = sum(1 for r in runs if r.passed)
        pass_color = "green" if n_passed == len(runs) else ("yellow" if n_passed > 0 else "red")
        label = f"{fx_name} [dim]({variant})[/dim]" if ab else fx_name
        aggregates.append(
            types_mod.FixtureAggregate(
                fixture_name=fx_name,
                variant=variant,
                runs=len(runs),
                passes=n_passed,
                dimensions={} if no_judge else _aggregate_dimensions(runs),
                hard_metrics=_aggregate_hard_metrics(runs),
            )
        )
        row = [label]
        if not no_judge:
            row.extend(_cell_for_runs(runs, dim) for dim, _ in _DIM_ORDER)
        row.append(f"[{pass_color}]{n_passed}/{len(runs)}[/{pass_color}]")
        table.add_row(*row)
    console.print(table)

    metrics_table = Table(show_header=True, header_style="bold", title="Operational metrics")
    metrics_table.add_column("Fixture / variant", no_wrap=True)
    metrics_table.add_column("Avg files", justify="right")
    metrics_table.add_column("Avg diff", justify="right")
    metrics_table.add_column("Avg tools", justify="right")
    metrics_table.add_column("Avg secs", justify="right")
    metrics_table.add_column("Verify pass", justify="right")
    for agg in aggregates:
        metrics = agg.hard_metrics
        label = f"{agg.fixture_name} ({agg.variant})" if ab else agg.fixture_name
        avg_diff = (
            f"+{metrics.get('avg_lines_added', 0):.1f}/-{metrics.get('avg_lines_deleted', 0):.1f}"
        )
        metrics_table.add_row(
            label,
            f"{metrics.get('avg_files_touched', 0):.1f}",
            avg_diff,
            f"{metrics.get('avg_tool_calls', 0):.1f}",
            f"{metrics.get('avg_total_duration_seconds', 0):.1f}",
            f"{metrics.get('verify_pass_rate', 0.0):.2f}",
        )
    console.print(metrics_table)

    defended_trials = [
        (ledger, passed) for ledger, passed, variant, _ in defense_trials if variant == "defended"
    ]
    if len(defended_trials) >= 3:
        stats = correlate_defenses(defended_trials)
        console.print()
        defense_table = Table(
            show_header=True,
            header_style="bold",
            title=f"Defense correlation ({len(defended_trials)} defended trials)",
        )
        defense_table.add_column("Defense", no_wrap=True)
        defense_table.add_column("block→pass", justify="center")
        defense_table.add_column("block→fail", justify="center")
        defense_table.add_column("silent→pass", justify="center")
        defense_table.add_column("silent→fail", justify="center")
        defense_table.add_column("Verdict", justify="left")
        verdict_color = {
            "helps": "green",
            "neutral": "yellow",
            "hurts": "red",
            "n/a": "dim",
            "n/a (small N)": "dim",
        }
        for s in stats:
            color = verdict_color.get(s.verdict(), "white")
            defense_table.add_row(
                s.name,
                str(s.block_pass),
                str(s.block_fail),
                str(s.silent_pass),
                str(s.silent_fail),
                f"[{color}]{s.verdict()}[/{color}]",
            )
        console.print(defense_table)
        console.print(
            "[dim]Read: a defense that 'hurts' fires when a trial fails more "
            "often than when it passes. Manually consider disabling such "
            "defenses; this report is diagnostic only.[/dim]"
        )

    if no_judge:
        report_data = {
            "run_id": run_id,
            "provider": resolved_provider,
            "model": resolved_model,
            "judge_provider": None,
            "judge_model": None,
            "fixture_set": fixture_set,
            "n_runs": n_runs,
            "ab": ab,
            "artifact_root": str(artifact_root),
            "benchmark_mode": benchmark_mode,
            "mutation_coverage": mutation_coverage,
            "no_judge": True,
            "results": [],
            "hard_results": hard_results,
            "aggregates": [agg.to_dict() for agg in aggregates],
        }
    else:
        report = types_mod.EvalReport(
            run_id=run_id,
            provider=resolved_provider,
            model=resolved_model,
            judge_provider=resolved_judge_provider,
            judge_model=resolved_judge_model,
            fixture_set=fixture_set,
            n_runs=n_runs,
            ab=ab,
            artifact_root=artifact_root,
            benchmark_mode=benchmark_mode,
            mutation_coverage=mutation_coverage,
            results=all_results,
            aggregates=aggregates,
        )
        report_data = report.to_dict()
    (artifact_root / "report.json").write_text(
        json.dumps(report_data, indent=2),
        encoding="utf-8",
    )
    if save_history:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report_data) + "\n")
    if json_out:
        console.print_json(json.dumps(report_data))
