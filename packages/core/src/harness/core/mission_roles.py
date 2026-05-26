from __future__ import annotations

from dataclasses import dataclass

from harness.core.mission_models import Mission

_DEFAULT_ROLE_BRIEFS: dict[str, str] = {
    "planner": (
        "Break the mission into milestones, features, and validation assertions "
        "before implementation starts."
    ),
    "worker": (
        "Implement the bounded feature against the handoff and validation contract, "
        "then leave a clean continuation note."
    ),
    "validator": (
        "Check milestone assertions independently, emit concrete findings, and block "
        "progress when evidence is incomplete."
    ),
    "reporter": (
        "Summarize current mission state, highlight blockers, and leave the next "
        "useful action for another agent or human."
    ),
}


@dataclass(frozen=True, slots=True)
class MissionRoleProfile:
    role: str
    model: str
    brief: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "model": self.model,
            "brief": self.brief,
        }


def resolve_mission_role_profile(*, mission: Mission, role: str) -> MissionRoleProfile:
    normalized = role.strip().lower() or "worker"
    model = ""
    brief = ""
    if normalized == "planner":
        model = mission.planner_model
        brief = mission.planner_brief
    elif normalized == "worker":
        model = mission.worker_model
        brief = mission.worker_brief
    elif normalized == "validator":
        model = mission.validator_model
        brief = mission.validator_brief
    elif normalized == "reporter":
        model = mission.reporter_model
        brief = mission.reporter_brief
    return MissionRoleProfile(
        role=normalized,
        model=model.strip(),
        brief=brief.strip() or _DEFAULT_ROLE_BRIEFS.get(normalized, ""),
    )


def mission_role_profiles(*, mission: Mission) -> tuple[MissionRoleProfile, ...]:
    return tuple(
        resolve_mission_role_profile(mission=mission, role=role)
        for role in ("planner", "worker", "validator", "reporter")
    )


__all__ = ["MissionRoleProfile", "mission_role_profiles", "resolve_mission_role_profile"]
