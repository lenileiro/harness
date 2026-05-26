from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from harness.core.mission_models import (
    Milestone,
    Mission,
    MissionFeature,
    MissionFinding,
    MissionHandoff,
    MissionRun,
    ValidationContract,
)


def default_mission_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve() / ".harness" / "missions"


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "item"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class MissionSearchHit:
    kind: str
    id: str
    title: str
    summary: str
    path: Path


class MissionStore:
    def __init__(self, *, root: Path):
        self.root = root

    @property
    def missions_dir(self) -> Path:
        return self.root / "missions"

    @property
    def milestones_dir(self) -> Path:
        return self.root / "milestones"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def contracts_dir(self) -> Path:
        return self.root / "contracts"

    @property
    def handoffs_dir(self) -> Path:
        return self.root / "handoffs"

    @property
    def findings_dir(self) -> Path:
        return self.root / "findings"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def ensure_layout(self) -> None:
        for path in (
            self.missions_dir,
            self.milestones_dir,
            self.features_dir,
            self.contracts_dir,
            self.handoffs_dir,
            self.findings_dir,
            self.runs_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def new_id(self, prefix: str, title: str) -> str:
        return f"{prefix}-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def add_mission(self, mission: Mission) -> Path:
        self.ensure_layout()
        target = self.missions_dir / mission.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "mission.json", mission.to_dict())
        markdown = [
            f"# {mission.title}",
            "",
            f"- id: `{mission.id}`",
            f"- status: `{mission.status}`",
            f"- created_by: `{mission.created_by}`",
            f"- created_at: `{mission.created_at}`",
            "",
            "## Goal",
            mission.goal,
        ]
        if mission.current_milestone_id:
            markdown += ["", f"- current_milestone_id: `{mission.current_milestone_id}`"]
        if mission.budget_tokens is not None or mission.budget_runtime_minutes is not None:
            markdown += ["", "## Budget"]
            if mission.budget_tokens is not None:
                markdown.append(f"- tokens: `{mission.budget_tokens}`")
            if mission.budget_runtime_minutes is not None:
                markdown.append(f"- runtime_minutes: `{mission.budget_runtime_minutes}`")
        role_models = {
            "planner_model": mission.planner_model,
            "worker_model": mission.worker_model,
            "validator_model": mission.validator_model,
            "reporter_model": mission.reporter_model,
        }
        if any(role_models.values()):
            markdown += ["", "## Role Models"]
            for label, value in role_models.items():
                if value:
                    markdown.append(f"- {label}: `{value}`")
        role_briefs = {
            "planner_brief": mission.planner_brief,
            "worker_brief": mission.worker_brief,
            "validator_brief": mission.validator_brief,
            "reporter_brief": mission.reporter_brief,
        }
        if any(role_briefs.values()):
            markdown += ["", "## Role Briefs"]
            for label, value in role_briefs.items():
                if value:
                    markdown.append(f"- {label}: {value}")
        (target / "MISSION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def update_mission(self, mission: Mission) -> Path:
        return self.add_mission(mission)

    def load_mission(self, mission_id: str) -> Mission:
        path = self.missions_dir / mission_id / "mission.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return Mission.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_missions(self, *, status: str | None = None) -> list[Mission]:
        if not self.missions_dir.exists():
            return []
        items: list[Mission] = []
        for path in sorted(self.missions_dir.iterdir()):
            payload = path / "mission.json"
            if not payload.is_file():
                continue
            mission = Mission.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if status and mission.status != status:
                continue
            items.append(mission)
        return items

    def add_milestone(self, milestone: Milestone) -> Path:
        self.ensure_layout()
        target = self.milestones_dir / milestone.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "milestone.json", milestone.to_dict())
        markdown = [
            f"# {milestone.title}",
            "",
            f"- id: `{milestone.id}`",
            f"- mission_id: `{milestone.mission_id}`",
            f"- status: `{milestone.status}`",
            f"- order: `{milestone.order}`",
            "",
            "## Summary",
            milestone.summary,
        ]
        (target / "MILESTONE.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def list_milestones(self, *, mission_id: str | None = None) -> list[Milestone]:
        if not self.milestones_dir.exists():
            return []
        items: list[Milestone] = []
        for path in sorted(self.milestones_dir.iterdir()):
            payload = path / "milestone.json"
            if not payload.is_file():
                continue
            milestone = Milestone.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and milestone.mission_id != mission_id:
                continue
            items.append(milestone)
        return items

    def load_milestone(self, milestone_id: str) -> Milestone:
        path = self.milestones_dir / milestone_id / "milestone.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return Milestone.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update_milestone(self, milestone: Milestone) -> Path:
        return self.add_milestone(milestone)

    def add_feature(self, feature: MissionFeature) -> Path:
        self.ensure_layout()
        target = self.features_dir / feature.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "feature.json", feature.to_dict())
        markdown = [
            f"# {feature.title}",
            "",
            f"- id: `{feature.id}`",
            f"- mission_id: `{feature.mission_id}`",
            f"- milestone_id: `{feature.milestone_id}`",
            f"- status: `{feature.status}`",
            f"- assigned_role: `{feature.assigned_role}`",
            "",
            "## Summary",
            feature.summary,
        ]
        if feature.depends_on:
            markdown += ["", "## Depends On", *[f"- `{item}`" for item in feature.depends_on]]
        if feature.target_files:
            markdown += ["", "## Target Files", *[f"- `{item}`" for item in feature.target_files]]
        if feature.research_refs:
            markdown += [
                "",
                "## Research References",
                *[f"- `{item}`" for item in feature.research_refs],
            ]
        (target / "FEATURE.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def list_features(
        self, *, mission_id: str | None = None, milestone_id: str | None = None
    ) -> list[MissionFeature]:
        if not self.features_dir.exists():
            return []
        items: list[MissionFeature] = []
        for path in sorted(self.features_dir.iterdir()):
            payload = path / "feature.json"
            if not payload.is_file():
                continue
            feature = MissionFeature.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and feature.mission_id != mission_id:
                continue
            if milestone_id and feature.milestone_id != milestone_id:
                continue
            items.append(feature)
        return items

    def load_feature(self, feature_id: str) -> MissionFeature:
        path = self.features_dir / feature_id / "feature.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return MissionFeature.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update_feature(self, feature: MissionFeature) -> Path:
        return self.add_feature(feature)

    def add_contract(self, contract: ValidationContract) -> Path:
        self.ensure_layout()
        target = self.contracts_dir / contract.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "contract.json", contract.to_dict())
        markdown = [
            f"# Validation Contract {contract.id}",
            "",
            f"- mission_id: `{contract.mission_id}`",
            f"- assertions: `{len(contract.assertions)}`",
            "",
            "## Summary",
            contract.summary,
        ]
        if contract.assertions:
            markdown += ["", "## Assertions"]
            for assertion in contract.assertions:
                markdown += [
                    f"- `{assertion.id}` {assertion.title}",
                    f"  - kind: `{assertion.kind}`",
                    f"  - verification_method: {assertion.verification_method}",
                ]
        (target / "CONTRACT.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def load_contract(self, contract_id: str) -> ValidationContract:
        path = self.contracts_dir / contract_id / "contract.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return ValidationContract.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_contracts(self, *, mission_id: str | None = None) -> list[ValidationContract]:
        if not self.contracts_dir.exists():
            return []
        items: list[ValidationContract] = []
        for path in sorted(self.contracts_dir.iterdir()):
            payload = path / "contract.json"
            if not payload.is_file():
                continue
            contract = ValidationContract.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and contract.mission_id != mission_id:
                continue
            items.append(contract)
        return items

    def load_contract_for_mission(self, mission_id: str) -> ValidationContract:
        contracts = self.list_contracts(mission_id=mission_id)
        if not contracts:
            raise FileNotFoundError(mission_id)
        return contracts[0]

    def add_handoff(self, handoff: MissionHandoff) -> Path:
        self.ensure_layout()
        target = self.handoffs_dir / handoff.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "handoff.json", handoff.to_dict())
        markdown = [
            f"# Handoff {handoff.id}",
            "",
            f"- mission_id: `{handoff.mission_id}`",
            f"- feature_id: `{handoff.feature_id}`",
            f"- role: `{handoff.role}`",
            f"- role_model: `{handoff.role_model}`",
            f"- confidence: `{handoff.confidence:.2f}`",
            "",
            "## Completed Work",
            handoff.completed_work,
        ]
        if handoff.remaining_work:
            markdown += ["", "## Remaining Work", handoff.remaining_work]
        if handoff.known_issues:
            markdown += ["", "## Known Issues", *[f"- {item}" for item in handoff.known_issues]]
        if handoff.next_recommendation:
            markdown += ["", "## Next Recommendation", handoff.next_recommendation]
        (target / "HANDOFF.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def load_handoff(self, handoff_id: str) -> MissionHandoff:
        path = self.handoffs_dir / handoff_id / "handoff.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return MissionHandoff.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_handoffs(
        self,
        *,
        mission_id: str | None = None,
        feature_id: str | None = None,
    ) -> list[MissionHandoff]:
        if not self.handoffs_dir.exists():
            return []
        items: list[MissionHandoff] = []
        for path in sorted(self.handoffs_dir.iterdir()):
            payload = path / "handoff.json"
            if not payload.is_file():
                continue
            handoff = MissionHandoff.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and handoff.mission_id != mission_id:
                continue
            if feature_id and handoff.feature_id != feature_id:
                continue
            items.append(handoff)
        return items

    def add_finding(self, finding: MissionFinding) -> Path:
        self.ensure_layout()
        target = self.findings_dir / finding.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "finding.json", finding.to_dict())
        markdown = [
            f"# Finding {finding.id}",
            "",
            f"- mission_id: `{finding.mission_id}`",
            f"- milestone_id: `{finding.milestone_id}`",
            f"- source: `{finding.source}`",
            f"- severity: `{finding.severity}`",
            "",
            "## Summary",
            finding.summary,
        ]
        if finding.recommended_fix:
            markdown += ["", "## Recommended Fix", finding.recommended_fix]
        (target / "FINDING.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def list_findings(
        self,
        *,
        mission_id: str | None = None,
        milestone_id: str | None = None,
    ) -> list[MissionFinding]:
        if not self.findings_dir.exists():
            return []
        items: list[MissionFinding] = []
        for path in sorted(self.findings_dir.iterdir()):
            payload = path / "finding.json"
            if not payload.is_file():
                continue
            finding = MissionFinding.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and finding.mission_id != mission_id:
                continue
            if milestone_id and finding.milestone_id != milestone_id:
                continue
            items.append(finding)
        return items

    def add_run(self, run: MissionRun) -> Path:
        self.ensure_layout()
        target = self.runs_dir / run.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "run.json", run.to_dict())
        markdown = [
            f"# Mission Run {run.id}",
            "",
            f"- mission_id: `{run.mission_id}`",
            f"- role: `{run.role}`",
            f"- role_model: `{run.role_model}`",
            f"- status: `{run.status}`",
            f"- related_feature_id: `{run.related_feature_id}`",
            f"- related_milestone_id: `{run.related_milestone_id}`",
        ]
        if run.summary:
            markdown += ["", "## Summary", run.summary]
        (target / "RUN.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def load_run(self, run_id: str) -> MissionRun:
        path = self.runs_dir / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return MissionRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(
        self,
        *,
        mission_id: str | None = None,
        role: str | None = None,
    ) -> list[MissionRun]:
        if not self.runs_dir.exists():
            return []
        items: list[MissionRun] = []
        for path in sorted(self.runs_dir.iterdir()):
            payload = path / "run.json"
            if not payload.is_file():
                continue
            run = MissionRun.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if mission_id and run.mission_id != mission_id:
                continue
            if role and run.role != role:
                continue
            items.append(run)
        return items


__all__ = ["MissionSearchHit", "MissionStore", "default_mission_root"]
