from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harness.core.mission_models import Milestone, Mission, MissionFeature, MissionFinding
from harness.core.mission_roles import mission_role_profiles
from harness.core.mission_store import MissionStore


@dataclass(frozen=True, slots=True)
class MissionSummaryReport:
    id: str
    mission_id: str
    status: str
    summary: str
    current_milestone_id: str
    role_profiles: tuple[dict[str, str], ...]
    milestone_statuses: tuple[dict[str, str], ...]
    feature_statuses: tuple[dict[str, str], ...]
    findings: tuple[dict[str, str], ...]
    next_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "status": self.status,
            "summary": self.summary,
            "current_milestone_id": self.current_milestone_id,
            "role_profiles": list(self.role_profiles),
            "milestone_statuses": list(self.milestone_statuses),
            "feature_statuses": list(self.feature_statuses),
            "findings": list(self.findings),
            "next_actions": list(self.next_actions),
        }


def _sorted_milestones(store: MissionStore, mission_id: str) -> list[Milestone]:
    return sorted(
        store.list_milestones(mission_id=mission_id),
        key=lambda item: (item.order, item.created_at, item.id),
    )


def _sorted_features(store: MissionStore, mission_id: str) -> list[MissionFeature]:
    return sorted(
        store.list_features(mission_id=mission_id),
        key=lambda item: (item.created_at, item.id),
    )


def _sorted_findings(store: MissionStore, mission_id: str) -> list[MissionFinding]:
    return sorted(
        store.list_findings(mission_id=mission_id),
        key=lambda item: (item.created_at, item.id),
        reverse=True,
    )


def _next_actions(
    mission: Mission,
    milestones: list[Milestone],
    features: list[MissionFeature],
    findings: list[MissionFinding],
) -> tuple[str, ...]:
    if mission.status == "completed":
        return ("Mission is complete.",)
    actions: list[str] = []
    if mission.status == "blocked":
        actions.append("Resolve validator findings before resuming execution.")
    current_milestone = next(
        (item for item in milestones if item.id == mission.current_milestone_id), None
    )
    if current_milestone is not None:
        current_features = [item for item in features if item.milestone_id == current_milestone.id]
        if any(
            item.status in {"pending", "handoff", "active", "blocked"} for item in current_features
        ):
            actions.append(f"Advance feature work in milestone '{current_milestone.title}'.")
        elif current_features and all(
            item.status in {"completed", "validated"} for item in current_features
        ):
            actions.append(f"Validate milestone '{current_milestone.title}'.")
    if findings:
        latest = findings[0]
        actions.append(f"Address latest {latest.severity} finding: {latest.summary}")
    return tuple(dict.fromkeys(actions))


def build_mission_summary_report(*, store: MissionStore, mission_id: str) -> MissionSummaryReport:
    mission = store.load_mission(mission_id)
    milestones = _sorted_milestones(store, mission_id)
    features = _sorted_features(store, mission_id)
    findings = _sorted_findings(store, mission_id)

    milestone_statuses = tuple(
        {
            "id": item.id,
            "title": item.title,
            "status": item.status,
        }
        for item in milestones
    )
    feature_statuses = tuple(
        {
            "id": item.id,
            "title": item.title,
            "status": item.status,
            "milestone_id": item.milestone_id,
        }
        for item in features
    )
    finding_rows = tuple(
        {
            "id": item.id,
            "severity": item.severity,
            "summary": item.summary,
            "recommended_fix": item.recommended_fix,
        }
        for item in findings[:5]
    )
    current_milestone = next(
        (item for item in milestones if item.id == mission.current_milestone_id), None
    )
    headline = (
        f"Mission '{mission.title}' is {mission.status}."
        if current_milestone is None
        else f"Mission '{mission.title}' is {mission.status} on milestone '{current_milestone.title}'."
    )
    return MissionSummaryReport(
        id=store.new_id("report", mission.title),
        mission_id=mission_id,
        status=mission.status,
        summary=headline,
        current_milestone_id=mission.current_milestone_id,
        role_profiles=tuple(
            profile.to_dict() for profile in mission_role_profiles(mission=mission)
        ),
        milestone_statuses=milestone_statuses,
        feature_statuses=feature_statuses,
        findings=finding_rows,
        next_actions=_next_actions(mission, milestones, features, findings),
    )


def write_mission_summary_report(
    *,
    store: MissionStore,
    report: MissionSummaryReport,
) -> Path:
    store.ensure_layout()
    target = store.reports_dir / report.id
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    lines = [
        f"# Mission Report {report.id}",
        "",
        f"- mission_id: `{report.mission_id}`",
        f"- status: `{report.status}`",
    ]
    if report.current_milestone_id:
        lines.append(f"- current_milestone_id: `{report.current_milestone_id}`")
    lines += ["", "## Summary", report.summary]
    if report.role_profiles:
        lines += ["", "## Role Profiles"]
        for item in report.role_profiles:
            label = item["role"]
            suffix = f" model=`{item['model']}`" if item["model"] else ""
            lines.append(f"- {label}{suffix}")
            if item["brief"]:
                lines.append(f"  - brief: {item['brief']}")
    if report.milestone_statuses:
        lines += ["", "## Milestones"]
        for item in report.milestone_statuses:
            lines.append(f"- `{item['id']}` {item['title']} [{item['status']}]")
    if report.feature_statuses:
        lines += ["", "## Features"]
        for item in report.feature_statuses:
            lines.append(
                f"- `{item['id']}` {item['title']} [{item['status']}] "
                f"(milestone `{item['milestone_id']}`)"
            )
    if report.findings:
        lines += ["", "## Findings"]
        for item in report.findings:
            lines.append(f"- `{item['severity']}` {item['summary']}")
            if item["recommended_fix"]:
                lines.append(f"  - fix: {item['recommended_fix']}")
    if report.next_actions:
        lines += ["", "## Next Actions"]
        for item in report.next_actions:
            lines.append(f"- {item}")
    (target / "REPORT.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


def list_mission_reports(
    *, store: MissionStore, mission_id: str | None = None
) -> list[MissionSummaryReport]:
    if not store.reports_dir.exists():
        return []
    items: list[MissionSummaryReport] = []
    for path in sorted(store.reports_dir.iterdir()):
        payload = path / "report.json"
        if not payload.is_file():
            continue
        data = json.loads(payload.read_text(encoding="utf-8"))
        report = MissionSummaryReport(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            status=str(data.get("status") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            current_milestone_id=str(data.get("current_milestone_id") or "").strip(),
            role_profiles=tuple(
                item for item in data.get("role_profiles") or [] if isinstance(item, dict)
            ),
            milestone_statuses=tuple(
                item for item in data.get("milestone_statuses") or [] if isinstance(item, dict)
            ),
            feature_statuses=tuple(
                item for item in data.get("feature_statuses") or [] if isinstance(item, dict)
            ),
            findings=tuple(item for item in data.get("findings") or [] if isinstance(item, dict)),
            next_actions=tuple(
                str(item).strip() for item in data.get("next_actions") or [] if str(item).strip()
            ),
        )
        if mission_id and report.mission_id != mission_id:
            continue
        items.append(report)
    return items


def load_mission_report(*, store: MissionStore, report_id: str) -> MissionSummaryReport:
    path = store.reports_dir / report_id / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return MissionSummaryReport(
        id=str(data["id"]),
        mission_id=str(data.get("mission_id") or "").strip(),
        status=str(data.get("status") or "").strip(),
        summary=str(data.get("summary") or "").strip(),
        current_milestone_id=str(data.get("current_milestone_id") or "").strip(),
        role_profiles=tuple(
            item for item in data.get("role_profiles") or [] if isinstance(item, dict)
        ),
        milestone_statuses=tuple(
            item for item in data.get("milestone_statuses") or [] if isinstance(item, dict)
        ),
        feature_statuses=tuple(
            item for item in data.get("feature_statuses") or [] if isinstance(item, dict)
        ),
        findings=tuple(item for item in data.get("findings") or [] if isinstance(item, dict)),
        next_actions=tuple(
            str(item).strip() for item in data.get("next_actions") or [] if str(item).strip()
        ),
    )


__all__ = [
    "MissionSummaryReport",
    "build_mission_summary_report",
    "list_mission_reports",
    "load_mission_report",
    "write_mission_summary_report",
]
