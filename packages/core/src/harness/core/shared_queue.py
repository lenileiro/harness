from __future__ import annotations

from dataclasses import dataclass

from harness.core.mission_models import MissionFeature
from harness.core.mission_store import MissionStore
from harness.core.research_scheduler import build_research_queue
from harness.core.research_store import ResearchStore

_DONE_FEATURE_STATUSES = {"completed", "validated"}
_ACTIVE_FEATURE_STATUSES = {"active", "handoff", "blocked"}


@dataclass(frozen=True, slots=True)
class SharedWorkItem:
    source: str
    kind: str
    id: str
    title: str
    summary: str
    status: str
    priority: int
    ready: bool
    owner_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "kind": self.kind,
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "priority": self.priority,
            "ready": self.ready,
            "owner_id": self.owner_id,
        }


def _sorted_mission_features(
    store: MissionStore, mission_id: str, milestone_id: str
) -> list[MissionFeature]:
    return sorted(
        store.list_features(mission_id=mission_id, milestone_id=milestone_id),
        key=lambda item: (item.created_at, item.id),
    )


def build_mission_work_queue(store: MissionStore) -> list[SharedWorkItem]:
    items: list[SharedWorkItem] = []
    missions = sorted(
        store.list_missions(),
        key=lambda item: (item.created_at, item.id),
    )
    for mission in missions:
        if mission.status not in {"approved", "running", "blocked"}:
            continue
        milestone_id = mission.current_milestone_id
        if not milestone_id:
            pending = sorted(
                (
                    item
                    for item in store.list_milestones(mission_id=mission.id)
                    if item.status != "completed"
                ),
                key=lambda item: (item.order, item.created_at, item.id),
            )
            if not pending:
                continue
            milestone_id = pending[0].id
        milestone = store.load_milestone(milestone_id)
        features = _sorted_mission_features(store, mission.id, milestone.id)
        if not features:
            items.append(
                SharedWorkItem(
                    source="mission",
                    kind="mission.validation",
                    id=milestone.id,
                    title=f"Validate {milestone.title}",
                    summary="Milestone has no features and can be validated immediately.",
                    status="ready",
                    priority=95,
                    ready=True,
                    owner_id=mission.id,
                )
            )
            continue

        active = [item for item in features if item.status in _ACTIVE_FEATURE_STATUSES]
        if active:
            feature = active[0]
            items.append(
                SharedWorkItem(
                    source="mission",
                    kind="mission.feature",
                    id=feature.id,
                    title=feature.title,
                    summary=feature.summary,
                    status=feature.status,
                    priority=100,
                    ready=False,
                    owner_id=mission.id,
                )
            )
            continue

        features_by_id = {item.id: item for item in features}
        ready_feature: MissionFeature | None = None
        blocked_feature: MissionFeature | None = None
        for feature in features:
            if feature.status in _DONE_FEATURE_STATUSES:
                continue
            unmet = [
                dependency
                for dependency in feature.depends_on
                if features_by_id.get(dependency) is None
                or features_by_id[dependency].status not in _DONE_FEATURE_STATUSES
            ]
            if unmet:
                blocked_feature = feature
                continue
            ready_feature = feature
            break

        if ready_feature is not None:
            items.append(
                SharedWorkItem(
                    source="mission",
                    kind="mission.feature",
                    id=ready_feature.id,
                    title=ready_feature.title,
                    summary=ready_feature.summary,
                    status="pending",
                    priority=90,
                    ready=True,
                    owner_id=mission.id,
                )
            )
            continue

        if all(item.status in _DONE_FEATURE_STATUSES for item in features):
            items.append(
                SharedWorkItem(
                    source="mission",
                    kind="mission.validation",
                    id=milestone.id,
                    title=f"Validate {milestone.title}",
                    summary="All feature work is complete; milestone validation is ready.",
                    status="ready_for_validation",
                    priority=95,
                    ready=True,
                    owner_id=mission.id,
                )
            )
            continue

        if blocked_feature is not None:
            items.append(
                SharedWorkItem(
                    source="mission",
                    kind="mission.feature",
                    id=blocked_feature.id,
                    title=blocked_feature.title,
                    summary=blocked_feature.summary,
                    status="blocked",
                    priority=70,
                    ready=False,
                    owner_id=mission.id,
                )
            )
    return sorted(items, key=lambda item: (-item.priority, item.source, item.id))


def build_research_work_queue(store: ResearchStore) -> list[SharedWorkItem]:
    return [
        SharedWorkItem(
            source="research",
            kind=f"research.{item.kind}",
            id=item.id,
            title=item.summary,
            summary=item.summary,
            status="ready",
            priority=item.priority,
            ready=True,
            owner_id=item.id,
        )
        for item in build_research_queue(store)
    ]


def build_shared_work_queue(
    *, mission_store: MissionStore, research_store: ResearchStore
) -> list[SharedWorkItem]:
    items = [*build_mission_work_queue(mission_store), *build_research_work_queue(research_store)]
    return sorted(items, key=lambda item: (-item.priority, item.source, item.id))


__all__ = [
    "SharedWorkItem",
    "build_mission_work_queue",
    "build_research_work_queue",
    "build_shared_work_queue",
]
