"""Stable extension protocols for runtime capabilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from harness.core.tool_entry import ToolSpec

if TYPE_CHECKING:
    from harness.core.approval import PendingApproval
    from harness.core.critic import Critic
    from harness.core.domain_profiles import DomainProfile
    from harness.core.gateway_models import GatewayMessage, GatewayReply
    from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord
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


@runtime_checkable
class LifecycleHook(Protocol):
    """Observer for scheduler and gateway runtime events."""

    def on_scheduler_tick(
        self,
        *,
        cwd: Path,
        started_at: datetime,
        finished_at: datetime,
        jobs_seen: int,
        jobs_executed: int,
        run_ids: tuple[str, ...],
    ) -> None: ...

    def on_job_started(
        self,
        *,
        cwd: Path,
        job: SchedulerJob,
        trigger: str,
        started_at: datetime,
    ) -> None: ...

    def on_job_completed(
        self,
        *,
        cwd: Path,
        job: SchedulerJob,
        trigger: str,
        record: SchedulerRunRecord,
    ) -> None: ...

    def on_gateway_message(
        self,
        *,
        cwd: Path,
        message: GatewayMessage,
    ) -> None: ...

    def on_gateway_reply(
        self,
        *,
        cwd: Path,
        message: GatewayMessage,
        reply: GatewayReply,
    ) -> None: ...

    def on_approval_requested(
        self,
        *,
        cwd: Path,
        approval: PendingApproval,
    ) -> None: ...

    def on_approval_resolved(
        self,
        *,
        cwd: Path,
        approval_id: str,
        granted: bool,
    ) -> None: ...


@runtime_checkable
class HookProvider(Protocol):
    """Factory source for one or more lifecycle hooks."""

    def hooks(self) -> list[LifecycleHook]: ...


__all__ = [
    "CriticProvider",
    "DomainProfileProvider",
    "ExperienceProvider",
    "HookProvider",
    "LifecycleHook",
    "ToolProvider",
    "VerifierProvider",
]
