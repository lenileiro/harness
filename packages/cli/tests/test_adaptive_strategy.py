from __future__ import annotations

from harness.cli.runtime_helpers import resolve_runtime_strategy


def test_adaptive_profile_uses_minimal_without_explicit_runtime_gates() -> None:
    strategy = resolve_runtime_strategy(
        prompt="# Add power method\nImplement power and tests.",
        requested_profile="adaptive",
        verify_command=None,
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "minimal"
    assert strategy.critic_mode is None


def test_explicit_phases_escalate_to_strict() -> None:
    strategy = resolve_runtime_strategy(
        prompt="Fix the bug.",
        requested_profile="adaptive",
        verify_command="pytest -q",
        phases="implement,test,verify",
        requested_critic=None,
    )
    assert strategy.structural_profile == "strict"
    assert strategy.critic_mode is None


def test_prompt_keywords_do_not_change_adaptive_strategy_without_explicit_gates() -> None:
    strategy = resolve_runtime_strategy(
        prompt=(
            "# Add power method\n"
            "Implement power and tests.\n\n"
            "Do not fix pre-existing typos, inconsistent formatting, or unused imports. "
            "Stay focused on the requested changes. Timeout, flaky, root cause, wrong layer."
        ),
        requested_profile="adaptive",
        verify_command=None,
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "minimal"
    assert strategy.critic_mode is None


def test_explicit_verify_command_enables_diagnostic_structure_without_implicit_critic() -> None:
    strategy = resolve_runtime_strategy(
        prompt="# Fix timeout bug\nThe real bug is likely downstream in concurrent request deduplication.",
        requested_profile="adaptive",
        verify_command="pytest tests/ -q",
        phases=None,
        requested_critic=None,
    )
    assert strategy.structural_profile == "diagnostic"
    assert strategy.critic_mode is None


def test_explicit_critic_is_preserved() -> None:
    strategy = resolve_runtime_strategy(
        prompt="Fix bug",
        requested_profile="adaptive",
        verify_command="pytest tests/ -q",
        phases=None,
        requested_critic="llm+search",
    )
    assert strategy.structural_profile == "diagnostic"
    assert strategy.critic_mode == "llm+search"


def test_explicit_profile_bypasses_adaptation() -> None:
    strategy = resolve_runtime_strategy(
        prompt="Fix bug",
        requested_profile="strict",
        verify_command="pytest -q",
        phases=None,
        requested_critic="llm+search",
    )
    assert strategy.structural_profile == "strict"
    assert strategy.critic_mode == "llm+search"
