"""Stable extension protocols for runtime capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from harness.core.tool_entry import ToolSpec

if TYPE_CHECKING:
    from harness.core.critic import Critic
    from harness.core.domain_profiles import DomainProfile
    from harness.core.tips_models import Tip
    from harness.core.verification import Verifier


@runtime_checkable
class ToolProvider(Protocol):
    """Declarative source of tool specs."""

    def specs(self) -> list[ToolSpec]: ...


@runtime_checkable
class VerifierProvider(Protocol):
    """Factory source for one or more verifier implementations."""

    def verifiers(self) -> list[Verifier]: ...


@runtime_checkable
class CriticProvider(Protocol):
    """Factory source for one or more critics."""

    def critics(self) -> list[Critic]: ...


@runtime_checkable
class ExperienceProvider(Protocol):
    """Source of reusable procedural guidance for a task."""

    def query(self, task_text: str, *, top_k: int = 3) -> list[Tip]: ...


@runtime_checkable
class DomainProfileProvider(Protocol):
    """Source of additional domain profiles."""

    def profiles(self) -> list[DomainProfile]: ...


__all__ = [
    "CriticProvider",
    "DomainProfileProvider",
    "ExperienceProvider",
    "ToolProvider",
    "VerifierProvider",
]
