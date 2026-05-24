from __future__ import annotations

from harness.cli import __main__ as cli_main


def test_feature_task_prefers_minimal_without_critic() -> None:
    strategy = cli_main._resolve_runtime_strategy(
        prompt="# Add power method\nImplement power and tests.",
        requested_profile="adaptive",
        verify_command="pytest -q",
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "minimal"
    assert strategy.critic_mode is None


def test_scope_sensitive_task_escalates_to_strict() -> None:
    strategy = cli_main._resolve_runtime_strategy(
        prompt="Fix the bug. Do not touch anything else. Minimal fix only.",
        requested_profile="adaptive",
        verify_command="pytest -q",
        phases="implement,test,verify",
        requested_critic=None,
    )
    assert strategy.structural_profile == "strict"


def test_feature_task_with_explicit_do_not_fix_constraint_escalates_to_strict() -> None:
    strategy = cli_main._resolve_runtime_strategy(
        prompt=(
            "# Add power method\n"
            "Implement power and tests.\n\n"
            "Do not fix pre-existing typos, inconsistent formatting, or unused imports. "
            "Stay focused on the requested changes."
        ),
        requested_profile="adaptive",
        verify_command="pytest -q",
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "strict"
    assert strategy.critic_mode is None


def test_diagnosis_heavy_bugfix_enables_critic() -> None:
    strategy = cli_main._resolve_runtime_strategy(
        prompt="# Fix timeout bug\nThe real bug is likely downstream in concurrent request deduplication.",
        requested_profile="adaptive",
        verify_command="pytest tests/ -q",
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "diagnostic"
    assert strategy.critic_mode == "llm"


def test_explicit_profile_bypasses_adaptation() -> None:
    strategy = cli_main._resolve_runtime_strategy(
        prompt="Fix bug",
        requested_profile="strict",
        verify_command="pytest -q",
        phases=None,
        requested_critic="llm+search",
    )
    assert strategy.structural_profile == "strict"
    assert strategy.critic_mode == "llm+search"
