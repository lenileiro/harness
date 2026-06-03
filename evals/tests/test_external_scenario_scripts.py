from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from evals.external_scenario_checks import is_generated_artifact, load_dotenv


@pytest.mark.parametrize(
    "script_name",
    [
        "external_js_scenario.py",
        "external_shell_scenario.py",
        "external_slug_scenario.py",
        "external_web_research_scenario.py",
    ],
)
def test_external_scenario_script_help_uses_stdlib_types(script_name: str) -> None:
    runner = Path(__file__).resolve().parents[1] / script_name

    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "scenario" in result.stdout.lower()
    assert "--env-file" in result.stdout


def test_external_scenario_dotenv_loader_sets_missing_values_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY='fake-openrouter-key'\n"
        'TAVILY_API_KEY="fake-tavily-key"\n'
        "EXISTING=value-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "already-set")

    load_dotenv(env_file)

    assert os.environ["OPENROUTER_API_KEY"] == "fake-openrouter-key"
    assert os.environ["TAVILY_API_KEY"] == "fake-tavily-key"
    assert os.environ["EXISTING"] == "already-set"


def test_external_scenario_generated_artifact_filter_is_language_neutral() -> None:
    assert is_generated_artifact("cache/blob.bin")
    assert is_generated_artifact("build/tool_cache/blob.bin")
    assert is_generated_artifact("work/.tool-cache/blob.bin")
    assert not is_generated_artifact("src/cacheable_feature.txt")


@pytest.mark.parametrize(
    "script_name",
    [
        "external_js_scenario.py",
        "external_shell_scenario.py",
        "external_slug_scenario.py",
        "external_web_research_scenario.py",
    ],
)
def test_external_scenarios_do_not_provide_prepared_default_verifiers(script_name: str) -> None:
    runner = Path(__file__).resolve().parents[1] / script_name
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "run_harness_on_external_environment"
    ]

    assert calls
    for call in calls:
        assert all(item.arg != "default_verify_command" for item in call.keywords)


def test_slug_scenario_natural_prompt_omits_explicit_verification_directions() -> None:
    from evals import external_slug_scenario as scenario

    prompt = scenario.scenario_instruction(natural_prompt=True).lower()

    assert "verify" not in prompt
    assert "verification" not in prompt
    assert "test" not in prompt
    assert "tool" not in prompt
    assert "slugify" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name",
    [
        "evals.external_js_scenario",
        "evals.external_shell_scenario",
        "evals.external_slug_scenario",
        "evals.external_web_research_scenario",
    ],
)
async def test_external_scenario_environment_replaces_invalid_utf8_output(
    module_name: str, tmp_path: Path
) -> None:
    module = __import__(module_name, fromlist=["LocalEnvironment"])
    env = module.LocalEnvironment(tmp_path)

    result = await env.exec("printf '\\341'")

    assert result.return_code == 0
    assert "\ufffd" in result.stdout


def test_web_research_scenario_accepts_added_focused_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import external_web_research_scenario as scenario

    workspace = scenario.create_workspace(tmp_path)
    (workspace / "current_python.py").write_text(
        """
CURRENT_PYTHON_RELEASE = "3.14.5"
SOURCE_URL = "https://www.python.org/downloads/release/python-3145/"


def current_python_release():
    return CURRENT_PYTHON_RELEASE


def source_url():
    return SOURCE_URL
""".lstrip(),
        encoding="utf-8",
    )
    (workspace / "tests" / "test_current_python_release_value.py").write_text(
        """
import current_python


def test_release_value():
    assert current_python.current_python_release() == "3.14.5"
    assert current_python.source_url().startswith("https://www.python.org/")
""".lstrip(),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    events = run_root / "harness" / "harness-events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "type": "tool_result",
                "result": {"name": "web_search", "is_error": False, "content": "ok"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scenario, "latest_python_release", lambda: "3.14.5")

    check = scenario.independent_check(workspace, run_root)

    assert check["changed_tests"] is True
    assert check["changed_test_paths"] == ["tests/test_current_python_release_value.py"]
    diff = check["diff"]
    assert isinstance(diff, str)
    assert "diff --git a/tests/test_current_python_release_value.py" in diff
    assert "+def test_release_value():" in diff
    assert check["leftover_scratch_paths"] == []


def test_js_scenario_accepts_added_focused_test(tmp_path: Path) -> None:
    from evals import external_js_scenario as scenario

    workspace = scenario.create_workspace(tmp_path)
    (workspace / "src" / "urlJoin.js").write_text(
        """
function joinUrl(...segments) {
  const cleaned = segments
    .filter((segment) => segment !== null && segment !== undefined && segment !== "")
    .map(String);
  if (cleaned.length === 0) return "";

  const origin = cleaned[0].match(/^(https?:\\/\\/[^/]+)(.*)$/);
  const pathSegments = origin ? [origin[2], ...cleaned.slice(1)] : cleaned;
  let path = pathSegments.join("/").replace(/\\/{2,}/g, "/");
  const isAbsolutePath = path.startsWith("/");
  if (path.length > 1) path = path.replace(/\\/+$/, "");
  if (path === "" && isAbsolutePath) path = "/";

  if (!origin) return path;
  if (path === "") return origin[1];
  if (path === "/") return `${origin[1]}/`;
  return origin[1] + (path.startsWith("/") ? path : `/${path}`);
}

module.exports = { joinUrl };
""".lstrip(),
        encoding="utf-8",
    )
    (workspace / "test" / "urlJoin.regression.test.js").write_text(
        """
const assert = require("node:assert/strict");
const test = require("node:test");
const { joinUrl } = require("../src/urlJoin");

test("preserves origin while ignoring empty segments", () => {
  assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");
});
""".lstrip(),
        encoding="utf-8",
    )

    check = scenario.independent_check(workspace)

    assert check["npm_test_return_code"] == 0
    assert check["behavior_return_code"] == 0
    assert check["changed_source"] is True
    assert check["changed_tests"] is True
    assert check["changed_test_paths"] == ["test/urlJoin.regression.test.js"]
    diff = check["diff"]
    assert isinstance(diff, str)
    assert "diff --git a/test/urlJoin.regression.test.js" in diff
    assert '+test("preserves origin while ignoring empty segments"' in diff
    assert check["leftover_scratch_paths"] == []


def test_shell_scenario_accepts_named_regression_script(tmp_path: Path) -> None:
    from evals import external_shell_scenario as scenario

    workspace = scenario.create_workspace(tmp_path)
    regression = workspace / "tests" / "ini_get_regression.sh"
    regression.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n", encoding="utf-8")
    regression.chmod(0o755)
    run_sh = workspace / "tests" / "run.sh"
    run_sh.write_text(
        run_sh.read_text(encoding="utf-8") + "\n./tests/ini_get_regression.sh\n", encoding="utf-8"
    )

    check = scenario.independent_check(workspace)

    assert check["changed_tests"] is True
    assert check["changed_test_paths"] == [
        "tests/run.sh",
        "tests/ini_get_regression.sh",
    ]
    diff = check["diff"]
    assert isinstance(diff, str)
    assert "diff --git a/tests/ini_get_regression.sh" in diff
    assert "+echo ok" in diff
    assert check["runner_wired_new_regression"] is True
    assert check["leftover_scratch_paths"] == []


def test_shell_scenario_hidden_check_preserves_internal_value_spaces(
    tmp_path: Path,
) -> None:
    from evals import external_shell_scenario as scenario

    workspace = scenario.create_workspace(tmp_path)
    (workspace / "bin" / "ini_get").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

section="$1"
key="$2"
file="$3"
current=""

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

while IFS= read -r line || [ -n "$line" ]; do
  if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*[#\\;] ]]; then
    continue
  fi
  if [[ "$line" =~ ^[[:space:]]*\\[[^\\]]*\\][[:space:]]*$ ]]; then
    inner="${line#*[}"
    inner="${inner%\\]*}"
    current="$(trim "$inner")"
    continue
  fi
  if [[ "$line" =~ ^[[:space:]]*([^=[:space:]]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
    found_key="$(trim "${BASH_REMATCH[1]}")"
    if [ "$current" = "$section" ] && [ "$found_key" = "$key" ]; then
      value="$(trim "${BASH_REMATCH[2]}")"
      printf '%s\n' "${value%% *}"
      exit 0
    fi
  fi
done < "$file"
exit 1
""",
        encoding="utf-8",
    )

    check = scenario.independent_check(workspace)

    behavior = check["behavior"]
    assert isinstance(behavior, list)
    description_case = next(
        item for item in behavior if isinstance(item, dict) and item.get("key") == "description"
    )
    stdout = description_case["stdout"]
    assert isinstance(stdout, str)
    assert description_case["expected_stdout"] == "value with spaces"
    assert stdout.strip() == "value"
    assert description_case["passed"] is False
    assert check["behavior_passed"] is False
