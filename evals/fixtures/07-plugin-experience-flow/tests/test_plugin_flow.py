from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(os.environ["HARNESS_EVAL_PROJECT_ROOT"]).resolve()


def _workspace() -> Path:
    return Path(os.environ["HARNESS_EVAL_WORKSPACE"]).resolve()


def _run_harness(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--project", str(_project_root()), "harness", *args],
        cwd=_workspace(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_workspace_experience_plugin_loads_and_queries() -> None:
    manifest = _workspace() / ".harness" / "plugins" / "workspace-experience.toml"
    provider_module = _workspace() / "workspace_plugin.py"

    assert manifest.exists(), "workspace plugin manifest was not created"
    assert provider_module.exists(), "workspace plugin module was not created"

    validate = _run_harness("plugins", "validate", "--kind", "experience", "--cwd", ".")
    assert validate.returncode == 0, validate.stdout + validate.stderr
    assert "workspace-experience" in validate.stdout
    assert "ok" in validate.stdout.lower()

    listed = _run_harness("plugins", "list", "--kind", "experience", "--cwd", ".")
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "workspace-experience" in listed.stdout

    provider_check = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(_project_root()),
            "python",
            "-c",
            (
                "import json; "
                "from pathlib import Path; "
                "from harness.cli.plugins import load_cli_experience_providers; "
                f"providers = load_cli_experience_providers(Path({str(_workspace())!r})); "
                "assert len(providers) == 1; "
                "tips = providers[0].query('plugin reliability check', top_k=5); "
                "other = providers[0].query('database migration', top_k=5); "
                "print(json.dumps({'tips': [tip.text for tip in tips], 'other': [tip.text for tip in other]}))"
            ),
        ],
        cwd=_workspace(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert provider_check.returncode == 0, provider_check.stdout + provider_check.stderr
    assert "Use plugin validation before runtime experiments." in provider_check.stdout
    assert '"other": []' in provider_check.stdout
