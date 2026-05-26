from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from harness.core.mission_models import (
    Milestone,
    Mission,
    MissionFeature,
    ValidationAssertion,
    ValidationContract,
)
from harness.core.mission_store import MissionStore


@dataclass(frozen=True, slots=True)
class PlannedMilestoneInput:
    label: str
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class PlannedAssertionInput:
    label: str
    title: str
    description: str
    kind: str
    verification_method: str


@dataclass(frozen=True, slots=True)
class PlannedFeatureInput:
    label: str
    milestone_label: str
    title: str
    summary: str
    assigned_role: str
    target_files: tuple[str, ...] = ()
    depends_on_labels: tuple[str, ...] = ()
    assertion_labels: tuple[str, ...] = ()
    research_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MissionPlanBundle:
    mission: Mission
    milestones: tuple[Milestone, ...]
    features: tuple[MissionFeature, ...]
    contract: ValidationContract


@dataclass(frozen=True, slots=True)
class MissionPlanDraft:
    contract_summary: str
    milestones: tuple[PlannedMilestoneInput, ...]
    assertions: tuple[PlannedAssertionInput, ...]
    features: tuple[PlannedFeatureInput, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_summary": self.contract_summary,
            "milestones": [
                {"label": item.label, "title": item.title, "summary": item.summary}
                for item in self.milestones
            ],
            "assertions": [
                {
                    "label": item.label,
                    "title": item.title,
                    "description": item.description,
                    "kind": item.kind,
                    "verification_method": item.verification_method,
                }
                for item in self.assertions
            ],
            "features": [
                {
                    "label": item.label,
                    "milestone_label": item.milestone_label,
                    "title": item.title,
                    "summary": item.summary,
                    "assigned_role": item.assigned_role,
                    "target_files": list(item.target_files),
                    "depends_on_labels": list(item.depends_on_labels),
                    "assertion_labels": list(item.assertion_labels),
                    "research_refs": list(item.research_refs),
                }
                for item in self.features
            ],
        }


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    body = text.strip()
    if body.startswith("```"):
        parts = body.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(body):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(body[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    return candidates


def parse_mission_plan_draft(text: str) -> MissionPlanDraft | None:
    for payload in reversed(_json_object_candidates(text)):
        contract_summary = str(payload.get("contract_summary") or "").strip()
        milestones_payload = payload.get("milestones") or []
        assertions_payload = payload.get("assertions") or []
        features_payload = payload.get("features") or []
        if not isinstance(milestones_payload, list):
            continue
        if not isinstance(assertions_payload, list):
            continue
        if not isinstance(features_payload, list):
            continue

        milestones: list[PlannedMilestoneInput] = []
        for item in milestones_payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not label or not title or not summary:
                continue
            milestones.append(PlannedMilestoneInput(label=label, title=title, summary=summary))

        assertions: list[PlannedAssertionInput] = []
        for item in assertions_payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
            kind = str(item.get("kind") or "").strip()
            verification_method = str(item.get("verification_method") or "").strip()
            if not label or not title or not description or not kind or not verification_method:
                continue
            assertions.append(
                PlannedAssertionInput(
                    label=label,
                    title=title,
                    description=description,
                    kind=kind,
                    verification_method=verification_method,
                )
            )

        features: list[PlannedFeatureInput] = []
        for item in features_payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            milestone_label = str(item.get("milestone_label") or "").strip()
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            assigned_role = str(item.get("assigned_role") or "worker").strip() or "worker"
            if not label or not milestone_label or not title or not summary:
                continue
            target_files = tuple(
                str(value).strip() for value in item.get("target_files") or [] if str(value).strip()
            )
            depends_on_labels = tuple(
                str(value).strip()
                for value in item.get("depends_on_labels") or []
                if str(value).strip()
            )
            assertion_labels = tuple(
                str(value).strip()
                for value in item.get("assertion_labels") or []
                if str(value).strip()
            )
            research_refs = tuple(
                str(value).strip()
                for value in item.get("research_refs") or []
                if str(value).strip()
            )
            if not assertion_labels:
                continue
            features.append(
                PlannedFeatureInput(
                    label=label,
                    milestone_label=milestone_label,
                    title=title,
                    summary=summary,
                    assigned_role=assigned_role,
                    target_files=target_files,
                    depends_on_labels=depends_on_labels,
                    assertion_labels=assertion_labels,
                    research_refs=research_refs,
                )
            )

        if contract_summary and milestones and assertions and features:
            return MissionPlanDraft(
                contract_summary=contract_summary,
                milestones=tuple(milestones),
                assertions=tuple(assertions),
                features=tuple(features),
            )
    return None


def build_mission_plan(
    *,
    store: MissionStore,
    mission: Mission,
    contract_summary: str,
    milestones: tuple[PlannedMilestoneInput, ...],
    assertions: tuple[PlannedAssertionInput, ...],
    features: tuple[PlannedFeatureInput, ...],
) -> MissionPlanBundle:
    if not milestones:
        raise ValueError("mission plan requires at least one milestone")
    if not assertions:
        raise ValueError("mission plan requires at least one validation assertion")
    if not features:
        raise ValueError("mission plan requires at least one feature")

    milestone_labels = {item.label for item in milestones}
    assertion_labels = {item.label for item in assertions}
    feature_labels = {item.label for item in features}
    if len(milestone_labels) != len(milestones):
        raise ValueError("mission plan milestone labels must be unique")
    if len(assertion_labels) != len(assertions):
        raise ValueError("mission plan assertion labels must be unique")
    if len(feature_labels) != len(features):
        raise ValueError("mission plan feature labels must be unique")

    contract = ValidationContract(
        id=store.new_id("contract", mission.title),
        mission_id=mission.id,
        summary=contract_summary.strip(),
    )
    assertion_map: dict[str, ValidationAssertion] = {}
    for item in assertions:
        assertion_map[item.label] = ValidationAssertion(
            id=store.new_id("assertion", item.title),
            contract_id=contract.id,
            title=item.title.strip(),
            description=item.description.strip(),
            kind=str(item.kind).strip(),  # type: ignore[arg-type]
            verification_method=item.verification_method.strip(),
        )

    milestone_map: dict[str, Milestone] = {}
    persisted_milestones: list[Milestone] = []
    for order, item in enumerate(milestones, start=1):
        milestone = Milestone(
            id=store.new_id("milestone", item.title),
            mission_id=mission.id,
            title=item.title.strip(),
            summary=item.summary.strip(),
            status="pending",
            order=order,
        )
        milestone_map[item.label] = milestone
        persisted_milestones.append(milestone)

    feature_id_by_label: dict[str, str] = {}
    for item in features:
        feature_id_by_label[item.label] = store.new_id("feature", item.title)

    covered_assertions: dict[str, list[str]] = {label: [] for label in assertion_labels}
    persisted_features: list[MissionFeature] = []
    for item in features:
        if item.milestone_label not in milestone_map:
            raise ValueError(
                f"feature {item.label!r} references unknown milestone {item.milestone_label!r}"
            )
        if not item.assertion_labels:
            raise ValueError(f"feature {item.label!r} must cover at least one assertion")
        unknown_assertions = [
            label for label in item.assertion_labels if label not in assertion_map
        ]
        if unknown_assertions:
            raise ValueError(
                f"feature {item.label!r} references unknown assertions: {', '.join(unknown_assertions)}"
            )
        unknown_dependencies = [
            label for label in item.depends_on_labels if label not in feature_id_by_label
        ]
        if unknown_dependencies:
            raise ValueError(
                f"feature {item.label!r} depends on unknown features: {', '.join(unknown_dependencies)}"
            )
        feature = MissionFeature(
            id=feature_id_by_label[item.label],
            mission_id=mission.id,
            milestone_id=milestone_map[item.milestone_label].id,
            title=item.title.strip(),
            summary=item.summary.strip(),
            status="pending",
            assigned_role=item.assigned_role.strip() or "worker",
            target_files=item.target_files,
            depends_on=tuple(feature_id_by_label[label] for label in item.depends_on_labels),
            research_refs=item.research_refs,
        )
        persisted_features.append(feature)
        for label in item.assertion_labels:
            covered_assertions[label].append(feature.id)

    missing_assertions = [
        label for label, covered_by in covered_assertions.items() if not covered_by
    ]
    if missing_assertions:
        raise ValueError(
            "mission plan assertions must be covered by at least one feature: "
            + ", ".join(missing_assertions)
        )

    persisted_assertions = tuple(
        replace(assertion, covered_by_features=tuple(covered_assertions[label]))
        for label, assertion in assertion_map.items()
    )
    persisted_contract = replace(contract, assertions=persisted_assertions)
    planned_mission = replace(
        mission,
        status="planned",
        current_milestone_id=persisted_milestones[0].id,
    )
    return MissionPlanBundle(
        mission=planned_mission,
        milestones=tuple(persisted_milestones),
        features=tuple(persisted_features),
        contract=persisted_contract,
    )


__all__ = [
    "MissionPlanBundle",
    "MissionPlanDraft",
    "PlannedAssertionInput",
    "PlannedFeatureInput",
    "PlannedMilestoneInput",
    "build_mission_plan",
    "parse_mission_plan_draft",
]
