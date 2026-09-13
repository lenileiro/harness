from typer.testing import CliRunner

from harness.cli.__main__ import app
from harness.core.experiment_plans import ExperimentPlan
from harness.core.research_store import ResearchStore, default_research_root


def test_failed_experiment_returns_failure_to_automation(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_experiment_plan(
        ExperimentPlan(id="failure", hypothesis_id="h", plan="Fail", checks=("exit 1",))
    )
    result = CliRunner().invoke(
        app, ["research", "experiment", "run", "--plan", "failure", "--cwd", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "failed" in result.stdout
