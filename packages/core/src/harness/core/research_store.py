from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from harness.core.citations import Citation
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiments import Experiment, ExperimentResult
from harness.core.hypotheses import Hypothesis
from harness.core.inspiration import InspirationNote
from harness.core.observations import Observation
from harness.core.opportunities import Opportunity
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.publications import ResearchAsset
from harness.core.research_archive import ArchivedResearchItem
from harness.core.research_models import (
    ChangeIntent,
    Publication,
    RabbitHole,
    Theme,
    Unknown,
    Vision,
)
from harness.core.section_maps import SectionMap


def default_research_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve() / ".harness" / "research"


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "item"


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ResearchSearchHit:
    kind: str
    id: str
    title: str
    summary: str
    path: Path


class ResearchStore:
    def __init__(self, *, root: Path):
        self.root = root

    @property
    def rabbit_holes_dir(self) -> Path:
        return self.root / "rabbitholes"

    @property
    def vision_dir(self) -> Path:
        return self.root / "vision" / "current"

    @property
    def themes_dir(self) -> Path:
        return self.root / "themes"

    @property
    def unknowns_dir(self) -> Path:
        return self.root / "unknowns"

    @property
    def publications_dir(self) -> Path:
        return self.root / "publications"

    @property
    def citations_dir(self) -> Path:
        return self.root / "citations"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def inspiration_dir(self) -> Path:
        return self.root / "inspiration"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def section_maps_dir(self) -> Path:
        return self.root / "section-maps"

    @property
    def observations_dir(self) -> Path:
        return self.root / "observations"

    @property
    def opportunities_dir(self) -> Path:
        return self.root / "opportunities"

    @property
    def hypotheses_dir(self) -> Path:
        return self.root / "hypotheses"

    @property
    def experiment_plans_dir(self) -> Path:
        return self.root / "experiment-plans"

    @property
    def promotion_candidates_dir(self) -> Path:
        return self.root / "promotions"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def ensure_layout(self) -> None:
        for path in (
            self.root / "vision",
            self.root / "themes",
            self.root / "unknowns",
            self.rabbit_holes_dir,
            self.publications_dir,
            self.citations_dir,
            self.inspiration_dir,
            self.assets_dir,
            self.section_maps_dir,
            self.observations_dir,
            self.opportunities_dir,
            self.hypotheses_dir,
            self.experiment_plans_dir,
            self.promotion_candidates_dir,
            self.experiments_dir,
            self.archive_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def new_id(self, prefix: str, title: str) -> str:
        return f"{prefix}-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def update_vision(self, vision: Vision) -> Path:
        self.ensure_layout()
        target = self.vision_dir
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "vision.json", vision.to_dict())
        markdown = [
            f"# {vision.title}",
            "",
            f"- id: `{vision.id}`",
            f"- updated_at: `{vision.updated_at}`",
            "",
            "## Summary",
            vision.summary,
        ]
        if vision.themes:
            markdown += ["", "## Themes", *[f"- {item}" for item in vision.themes]]
        if vision.success_metrics:
            markdown += [
                "",
                "## Success Metrics",
                *[f"- {item}" for item in vision.success_metrics],
            ]
        (target / "VISION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_theme(self, theme: Theme) -> Path:
        self.ensure_layout()
        target = self.themes_dir / theme.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "theme.json", theme.to_dict())
        markdown = [
            f"# {theme.title}",
            "",
            f"- id: `{theme.id}`",
            f"- vision_id: `{theme.vision_id}`",
            f"- priority: `{theme.priority}`",
            f"- status: `{theme.status}`",
            "",
            "## Description",
            theme.description,
        ]
        (target / "THEME.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_unknown(self, unknown: Unknown) -> Path:
        self.ensure_layout()
        target = self.unknowns_dir / unknown.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "unknown.json", unknown.to_dict())
        markdown = [
            f"# {unknown.question}",
            "",
            f"- id: `{unknown.id}`",
            f"- theme_id: `{unknown.theme_id}`",
            f"- status: `{unknown.status}`",
            f"- confidence: `{unknown.confidence:.2f}`",
            f"- created_by: `{unknown.created_by}`",
            "",
            "## Why It Matters",
            unknown.why_it_matters,
        ]
        if unknown.current_belief:
            markdown += ["", "## Current Belief", unknown.current_belief]
        if unknown.related_sections:
            markdown += [
                "",
                "## Related Sections",
                *[f"- {item}" for item in unknown.related_sections],
            ]
        (target / "UNKNOWN.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_rabbit_hole(self, rabbit_hole: RabbitHole) -> Path:
        self.ensure_layout()
        target = self.rabbit_holes_dir / rabbit_hole.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "rabbit_hole.json", rabbit_hole.to_dict())
        markdown = [
            f"# {rabbit_hole.title}",
            "",
            f"- id: `{rabbit_hole.id}`",
            f"- theme: `{rabbit_hole.theme}`",
            f"- status: `{rabbit_hole.status}`",
            f"- opened_by: `{rabbit_hole.opened_by}`",
            f"- created_at: `{rabbit_hole.created_at}`",
            "",
            "## Question",
            rabbit_hole.question,
            "",
            "## Scope",
            rabbit_hole.scope,
        ]
        if rabbit_hole.related_sections:
            markdown += [
                "",
                "## Related Sections",
                *[f"- {item}" for item in rabbit_hole.related_sections],
            ]
        if rabbit_hole.tags:
            markdown += ["", "## Tags", *[f"- {item}" for item in rabbit_hole.tags]]
        if rabbit_hole.change_intent is not None:
            markdown += [
                "",
                "## Change Intent",
                f"- mode: `{rabbit_hole.change_intent.mode}`",
                f"- subsystem: `{rabbit_hole.change_intent.subsystem}`",
                f"- risk: `{rabbit_hole.change_intent.risk}`",
                "",
                rabbit_hole.change_intent.rationale,
                "",
                f"Expected outcome: {rabbit_hole.change_intent.expected_outcome}",
            ]
        (target / "RABBITHOLE.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_publication(self, publication: Publication) -> Path:
        self.ensure_layout()
        target = self.publications_dir / publication.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "publication.json", publication.to_dict())
        markdown = [
            f"# {publication.title}",
            "",
            f"- id: `{publication.id}`",
            f"- rabbit_hole_id: `{publication.rabbit_hole_id}`",
            f"- status: `{publication.status}`",
            f"- confidence: `{publication.confidence:.1f}`",
            f"- created_at: `{publication.created_at}`",
            "",
            "## Summary",
            publication.summary,
        ]
        if publication.claims:
            markdown += ["", "## Claims", *[f"- {item}" for item in publication.claims]]
        if publication.supporting_evidence:
            markdown += [
                "",
                "## Supporting Evidence",
                *[f"- {item}" for item in publication.supporting_evidence],
            ]
        if publication.counterevidence:
            markdown += [
                "",
                "## Counterevidence",
                *[f"- {item}" for item in publication.counterevidence],
            ]
        if publication.recommendations:
            markdown += [
                "",
                "## Recommendations",
                *[f"- {item}" for item in publication.recommendations],
            ]
        if publication.open_questions:
            markdown += [
                "",
                "## Open Questions",
                *[f"- {item}" for item in publication.open_questions],
            ]
        if publication.sources:
            markdown += ["", "## Sources", *[f"- {item}" for item in publication.sources]]
        if publication.artifacts:
            markdown += ["", "## Artifacts", *[f"- {item}" for item in publication.artifacts]]
        if publication.citations:
            markdown += ["", "## Citations", *[f"- {item}" for item in publication.citations]]
        if publication.change_intent is not None:
            markdown += [
                "",
                "## Change Intent",
                f"- mode: `{publication.change_intent.mode}`",
                f"- subsystem: `{publication.change_intent.subsystem}`",
                f"- risk: `{publication.change_intent.risk}`",
                "",
                publication.change_intent.rationale,
                "",
                f"Expected outcome: {publication.change_intent.expected_outcome}",
            ]
        (target / "PUBLICATION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_inspiration_note(self, note: InspirationNote) -> Path:
        self.ensure_layout()
        target = self.inspiration_dir / note.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "inspiration.json", note.to_dict())
        markdown = [
            f"# {note.title}",
            "",
            f"- id: `{note.id}`",
            f"- source_kind: `{note.source.kind}`",
            f"- source_ref: `{note.source.ref}`",
            f"- created_by: `{note.created_by}`",
            "",
            "## Summary",
            note.summary,
        ]
        if note.source.title or note.source.excerpt:
            markdown += ["", "## Source"]
            if note.source.title:
                markdown.append(f"- title: {note.source.title}")
            if note.source.excerpt:
                markdown.append(f"- excerpt: {note.source.excerpt}")
        if note.related_themes:
            markdown += ["", "## Related Themes", *[f"- {item}" for item in note.related_themes]]
        if note.related_sections:
            markdown += [
                "",
                "## Related Sections",
                *[f"- {item}" for item in note.related_sections],
            ]
        (target / "INSPIRATION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_citation(self, citation: Citation) -> Path:
        self.ensure_layout()
        target = self.citations_dir / citation.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "citation.json", citation.to_dict())
        markdown = [
            f"# {citation.id}",
            "",
            f"- source_publication_id: `{citation.source_publication_id}`",
            f"- target_publication_id: `{citation.target_publication_id}`",
            f"- relationship: `{citation.relationship}`",
        ]
        if citation.note:
            markdown += ["", "## Note", citation.note]
        (target / "CITATION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_research_asset(self, asset: ResearchAsset) -> Path:
        self.ensure_layout()
        target = self.assets_dir / asset.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "asset.json", asset.to_dict())
        return target

    def add_section_map(self, section_map: SectionMap) -> Path:
        self.ensure_layout()
        target = self.section_maps_dir / section_map.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "section_map.json", section_map.to_dict())
        markdown = [
            f"# {section_map.section}",
            "",
            f"- id: `{section_map.id}`",
            f"- created_by: `{section_map.created_by}`",
            f"- created_at: `{section_map.created_at}`",
        ]
        if section_map.files:
            markdown += ["", "## Files", *[f"- {item}" for item in section_map.files]]
        if section_map.interfaces:
            markdown += ["", "## Interfaces", *[f"- {item}" for item in section_map.interfaces]]
        if section_map.constraints:
            markdown += ["", "## Constraints", *[f"- {item}" for item in section_map.constraints]]
        if section_map.weaknesses:
            markdown += ["", "## Weaknesses", *[f"- {item}" for item in section_map.weaknesses]]
        if section_map.opportunities:
            markdown += [
                "",
                "## Opportunities",
                *[f"- {item}" for item in section_map.opportunities],
            ]
        (target / "SECTION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_observation(self, observation: Observation) -> Path:
        self.ensure_layout()
        target = self.observations_dir / observation.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "observation.json", observation.to_dict())
        markdown = [
            f"# {observation.title}",
            "",
            f"- id: `{observation.id}`",
            f"- source_type: `{observation.source_type}`",
            f"- source_ref: `{observation.source_ref or '—'}`",
            f"- theme: `{observation.theme or '—'}`",
            f"- created_by: `{observation.created_by}`",
            f"- created_at: `{observation.created_at}`",
            "",
            "## Summary",
            observation.summary,
        ]
        if observation.related_sections:
            markdown += [
                "",
                "## Related Sections",
                *[f"- {item}" for item in observation.related_sections],
            ]
        (target / "OBSERVATION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_opportunity(self, opportunity: Opportunity) -> Path:
        self.ensure_layout()
        target = self.opportunities_dir / opportunity.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "opportunity.json", opportunity.to_dict())
        markdown = [
            f"# {opportunity.title}",
            "",
            f"- id: `{opportunity.id}`",
            f"- theme: `{opportunity.theme or '—'}`",
            *([f"- mission_id: `{opportunity.mission_id}`"] if opportunity.mission_id else []),
            *(
                [f"- mission_feature_id: `{opportunity.mission_feature_id}`"]
                if opportunity.mission_feature_id
                else []
            ),
            f"- priority: `{opportunity.priority}`",
            f"- created_by: `{opportunity.created_by}`",
            f"- created_at: `{opportunity.created_at}`",
            "",
            "## Summary",
            opportunity.summary,
        ]
        if opportunity.related_sections:
            markdown += [
                "",
                "## Related Sections",
                *[f"- {item}" for item in opportunity.related_sections],
            ]
        if opportunity.origin_observations:
            markdown += [
                "",
                "## Origin Observations",
                *[f"- {item}" for item in opportunity.origin_observations],
            ]
        if opportunity.change_modes:
            markdown += ["", "## Change Modes", *[f"- {item}" for item in opportunity.change_modes]]
        (target / "OPPORTUNITY.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_hypothesis(self, hypothesis: Hypothesis) -> Path:
        self.ensure_layout()
        target = self.hypotheses_dir / hypothesis.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "hypothesis.json", hypothesis.to_dict())
        markdown = [
            f"# {hypothesis.id}",
            "",
            f"- opportunity_id: `{hypothesis.opportunity_id}`",
            *([f"- mission_id: `{hypothesis.mission_id}`"] if hypothesis.mission_id else []),
            *(
                [f"- mission_feature_id: `{hypothesis.mission_feature_id}`"]
                if hypothesis.mission_feature_id
                else []
            ),
            f"- change_mode: `{hypothesis.change_mode}`",
            f"- risk_level: `{hypothesis.risk_level}`",
            f"- created_by: `{hypothesis.created_by}`",
            f"- created_at: `{hypothesis.created_at}`",
            "",
            "## Claim",
            hypothesis.claim,
            "",
            "## Expected Win",
            hypothesis.expected_win,
        ]
        (target / "HYPOTHESIS.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_experiment_plan(self, experiment_plan: ExperimentPlan) -> Path:
        self.ensure_layout()
        target = self.experiment_plans_dir / experiment_plan.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "experiment_plan.json", experiment_plan.to_dict())
        markdown = [
            f"# {experiment_plan.id}",
            "",
            f"- hypothesis_id: `{experiment_plan.hypothesis_id}`",
            f"- created_by: `{experiment_plan.created_by}`",
            f"- created_at: `{experiment_plan.created_at}`",
            "",
            "## Plan",
            experiment_plan.plan,
        ]
        if experiment_plan.target_files:
            markdown += [
                "",
                "## Target Files",
                *[f"- {item}" for item in experiment_plan.target_files],
            ]
        if experiment_plan.checks:
            markdown += ["", "## Checks", *[f"- {item}" for item in experiment_plan.checks]]
        if experiment_plan.eval_slices:
            markdown += [
                "",
                "## Eval Slices",
                *[f"- {item}" for item in experiment_plan.eval_slices],
            ]
        (target / "EXPERIMENT_PLAN.md").write_text(
            "\n".join(markdown).strip() + "\n", encoding="utf-8"
        )
        return target

    def add_promotion_candidate(self, candidate: PromotionCandidate) -> Path:
        self.ensure_layout()
        target = self.promotion_candidates_dir / candidate.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "promotion_candidate.json", candidate.to_dict())
        markdown = [
            f"# {candidate.title}",
            "",
            f"- id: `{candidate.id}`",
            *([f"- mission_id: `{candidate.mission_id}`"] if candidate.mission_id else []),
            f"- risk_level: `{candidate.risk_level}`",
            f"- created_by: `{candidate.created_by}`",
            f"- created_at: `{candidate.created_at}`",
            "",
            "## Summary",
            candidate.summary,
        ]
        if candidate.target_files:
            markdown += ["", "## Target Files", *[f"- {item}" for item in candidate.target_files]]
        if candidate.expected_metric:
            markdown += ["", "## Expected Metric", candidate.expected_metric]
        if candidate.validation_plan:
            markdown += ["", "## Validation Plan", candidate.validation_plan]
        if candidate.source_publications:
            markdown += [
                "",
                "## Source Publications",
                *[f"- {item}" for item in candidate.source_publications],
            ]
        if candidate.mission_feature_ids:
            markdown += [
                "",
                "## Mission Features",
                *[f"- {item}" for item in candidate.mission_feature_ids],
            ]
        if candidate.source_hypotheses:
            markdown += [
                "",
                "## Source Hypotheses",
                *[f"- {item}" for item in candidate.source_hypotheses],
            ]
        if candidate.change_intent is not None:
            markdown += [
                "",
                "## Change Intent",
                f"- mode: `{candidate.change_intent.mode}`",
                f"- subsystem: `{candidate.change_intent.subsystem}`",
                f"- risk: `{candidate.change_intent.risk}`",
                "",
                candidate.change_intent.rationale,
                "",
                f"Expected outcome: {candidate.change_intent.expected_outcome}",
            ]
        (target / "PROMOTION.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
        return target

    def add_experiment(self, experiment: Experiment, result: ExperimentResult) -> Path:
        self.ensure_layout()
        target = self.experiments_dir / experiment.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "experiment.json", experiment.to_dict())
        _write_json(target / "result.json", result.to_dict())
        return target

    def archive_item(self, *, kind: str, item_id: str, reason: str, note: str = "") -> Path:
        self.ensure_layout()
        source = self._source_path_for_kind(kind, item_id)
        if not source.exists():
            raise FileNotFoundError(f"unknown {kind}: {item_id}")
        archive_id = self.new_id("archive", f"{kind}-{item_id}")
        target = self.archive_dir / archive_id
        target.mkdir(parents=True, exist_ok=False)
        entry = ArchivedResearchItem(
            archive_id=archive_id,
            kind=kind,
            original_id=item_id,
            original_relpath=str(source.relative_to(self.root)),
            reason=reason.strip(),
            note=note.strip(),
        )
        _write_json(target / "archive.json", entry.to_dict())
        shutil.move(str(source), str(target / "artifact"))
        return target

    def load_rabbit_hole(self, rabbit_hole_id: str) -> RabbitHole:
        payload = json.loads(
            (self.rabbit_holes_dir / rabbit_hole_id / "rabbit_hole.json").read_text(
                encoding="utf-8"
            )
        )
        return RabbitHole.from_dict(payload)

    def load_vision(self) -> Vision:
        payload = json.loads((self.vision_dir / "vision.json").read_text(encoding="utf-8"))
        return Vision.from_dict(payload)

    def load_theme(self, theme_id: str) -> Theme:
        payload = json.loads(
            (self.themes_dir / theme_id / "theme.json").read_text(encoding="utf-8")
        )
        return Theme.from_dict(payload)

    def load_unknown(self, unknown_id: str) -> Unknown:
        payload = json.loads(
            (self.unknowns_dir / unknown_id / "unknown.json").read_text(encoding="utf-8")
        )
        return Unknown.from_dict(payload)

    def load_publication(self, publication_id: str) -> Publication:
        payload = json.loads(
            (self.publications_dir / publication_id / "publication.json").read_text(
                encoding="utf-8"
            )
        )
        return Publication.from_dict(payload)

    def load_citation(self, citation_id: str) -> Citation:
        payload = json.loads(
            (self.citations_dir / citation_id / "citation.json").read_text(encoding="utf-8")
        )
        return Citation.from_dict(payload)

    def load_research_asset(self, asset_id: str) -> ResearchAsset:
        payload = json.loads(
            (self.assets_dir / asset_id / "asset.json").read_text(encoding="utf-8")
        )
        return ResearchAsset.from_dict(payload)

    def load_inspiration_note(self, inspiration_id: str) -> InspirationNote:
        payload = json.loads(
            (self.inspiration_dir / inspiration_id / "inspiration.json").read_text(encoding="utf-8")
        )
        return InspirationNote.from_dict(payload)

    def load_section_map(self, section_map_id: str) -> SectionMap:
        payload = json.loads(
            (self.section_maps_dir / section_map_id / "section_map.json").read_text(
                encoding="utf-8"
            )
        )
        return SectionMap.from_dict(payload)

    def load_observation(self, observation_id: str) -> Observation:
        payload = json.loads(
            (self.observations_dir / observation_id / "observation.json").read_text(
                encoding="utf-8"
            )
        )
        return Observation.from_dict(payload)

    def load_opportunity(self, opportunity_id: str) -> Opportunity:
        payload = json.loads(
            (self.opportunities_dir / opportunity_id / "opportunity.json").read_text(
                encoding="utf-8"
            )
        )
        return Opportunity.from_dict(payload)

    def load_hypothesis(self, hypothesis_id: str) -> Hypothesis:
        payload = json.loads(
            (self.hypotheses_dir / hypothesis_id / "hypothesis.json").read_text(encoding="utf-8")
        )
        return Hypothesis.from_dict(payload)

    def load_experiment_plan(self, plan_id: str) -> ExperimentPlan:
        payload = json.loads(
            (self.experiment_plans_dir / plan_id / "experiment_plan.json").read_text(
                encoding="utf-8"
            )
        )
        return ExperimentPlan.from_dict(payload)

    def load_promotion_candidate(self, candidate_id: str) -> PromotionCandidate:
        payload = json.loads(
            (self.promotion_candidates_dir / candidate_id / "promotion_candidate.json").read_text(
                encoding="utf-8"
            )
        )
        return PromotionCandidate.from_dict(payload)

    def load_experiment(self, experiment_id: str) -> Experiment:
        payload = json.loads(
            (self.experiments_dir / experiment_id / "experiment.json").read_text(encoding="utf-8")
        )
        return Experiment.from_dict(payload)

    def load_experiment_result(self, experiment_id: str) -> ExperimentResult:
        payload = json.loads(
            (self.experiments_dir / experiment_id / "result.json").read_text(encoding="utf-8")
        )
        return ExperimentResult.from_dict(payload)

    def load_archive_item(self, archive_id: str) -> ArchivedResearchItem:
        payload = json.loads(
            (self.archive_dir / archive_id / "archive.json").read_text(encoding="utf-8")
        )
        return ArchivedResearchItem.from_dict(payload)

    def find_section_map(self, section_or_id: str) -> SectionMap | None:
        if (self.section_maps_dir / section_or_id / "section_map.json").is_file():
            return self.load_section_map(section_or_id)
        if not self.section_maps_dir.exists():
            return None
        needle = section_or_id.lower().strip()
        for entry in sorted(self.section_maps_dir.iterdir()):
            json_path = entry / "section_map.json"
            if not json_path.is_file():
                continue
            section_map = SectionMap.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if section_map.section.lower() == needle:
                return section_map
        return None

    def list_themes(self, *, vision_id: str | None = None) -> list[Theme]:
        if not self.themes_dir.exists():
            return []
        items: list[Theme] = []
        for entry in sorted(self.themes_dir.iterdir()):
            json_path = entry / "theme.json"
            if not json_path.is_file():
                continue
            theme = Theme.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if vision_id and theme.vision_id != vision_id:
                continue
            items.append(theme)
        return items

    def list_unknowns(
        self, *, theme_id: str | None = None, status: str | None = None
    ) -> list[Unknown]:
        if not self.unknowns_dir.exists():
            return []
        items: list[Unknown] = []
        for entry in sorted(self.unknowns_dir.iterdir()):
            json_path = entry / "unknown.json"
            if not json_path.is_file():
                continue
            unknown = Unknown.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if theme_id and unknown.theme_id != theme_id:
                continue
            if status and unknown.status != status:
                continue
            items.append(unknown)
        return items

    def list_rabbit_holes(self) -> list[RabbitHole]:
        if not self.rabbit_holes_dir.exists():
            return []
        items: list[RabbitHole] = []
        for entry in sorted(self.rabbit_holes_dir.iterdir()):
            json_path = entry / "rabbit_hole.json"
            if not json_path.is_file():
                continue
            items.append(RabbitHole.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
        return items

    def list_publications(self) -> list[Publication]:
        if not self.publications_dir.exists():
            return []
        items: list[Publication] = []
        for entry in sorted(self.publications_dir.iterdir()):
            json_path = entry / "publication.json"
            if not json_path.is_file():
                continue
            items.append(Publication.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
        return items

    def list_citations(self) -> list[Citation]:
        if not self.citations_dir.exists():
            return []
        items: list[Citation] = []
        for entry in sorted(self.citations_dir.iterdir()):
            json_path = entry / "citation.json"
            if not json_path.is_file():
                continue
            items.append(Citation.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
        return items

    def list_research_assets(self) -> list[ResearchAsset]:
        if not self.assets_dir.exists():
            return []
        items: list[ResearchAsset] = []
        for entry in sorted(self.assets_dir.iterdir()):
            json_path = entry / "asset.json"
            if not json_path.is_file():
                continue
            items.append(ResearchAsset.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
        return items

    def list_inspiration_notes(self, *, source_kind: str | None = None) -> list[InspirationNote]:
        if not self.inspiration_dir.exists():
            return []
        items: list[InspirationNote] = []
        for entry in sorted(self.inspiration_dir.iterdir()):
            json_path = entry / "inspiration.json"
            if not json_path.is_file():
                continue
            note = InspirationNote.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if source_kind and note.source.kind != source_kind:
                continue
            items.append(note)
        return items

    def list_opportunities(self, *, theme: str | None = None) -> list[Opportunity]:
        if not self.opportunities_dir.exists():
            return []
        items: list[Opportunity] = []
        for entry in sorted(self.opportunities_dir.iterdir()):
            json_path = entry / "opportunity.json"
            if not json_path.is_file():
                continue
            opportunity = Opportunity.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if theme and opportunity.theme != theme:
                continue
            items.append(opportunity)
        return items

    def related_opportunities(self, target: str) -> list[Opportunity]:
        needle = target.lower().strip()
        matches: list[Opportunity] = []
        for opportunity in self.list_opportunities():
            haystack = " ".join(
                (
                    opportunity.title,
                    opportunity.summary,
                    " ".join(opportunity.related_sections),
                    " ".join(opportunity.origin_observations),
                    opportunity.theme,
                )
            ).lower()
            if needle in haystack:
                matches.append(opportunity)
        return matches

    def list_hypotheses(self, *, opportunity_id: str | None = None) -> list[Hypothesis]:
        if not self.hypotheses_dir.exists():
            return []
        items: list[Hypothesis] = []
        for entry in sorted(self.hypotheses_dir.iterdir()):
            json_path = entry / "hypothesis.json"
            if not json_path.is_file():
                continue
            hypothesis = Hypothesis.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if opportunity_id and hypothesis.opportunity_id != opportunity_id:
                continue
            items.append(hypothesis)
        return items

    def list_experiment_plans(self, *, hypothesis_id: str | None = None) -> list[ExperimentPlan]:
        if not self.experiment_plans_dir.exists():
            return []
        items: list[ExperimentPlan] = []
        for entry in sorted(self.experiment_plans_dir.iterdir()):
            json_path = entry / "experiment_plan.json"
            if not json_path.is_file():
                continue
            plan = ExperimentPlan.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            if hypothesis_id and plan.hypothesis_id != hypothesis_id:
                continue
            items.append(plan)
        return items

    def list_promotion_candidates(self) -> list[PromotionCandidate]:
        if not self.promotion_candidates_dir.exists():
            return []
        items: list[PromotionCandidate] = []
        for entry in sorted(self.promotion_candidates_dir.iterdir()):
            json_path = entry / "promotion_candidate.json"
            if not json_path.is_file():
                continue
            candidate = PromotionCandidate.from_dict(
                json.loads(json_path.read_text(encoding="utf-8"))
            )
            items.append(candidate)
        return items

    def list_experiments(self) -> list[Experiment]:
        if not self.experiments_dir.exists():
            return []
        items: list[Experiment] = []
        for entry in sorted(self.experiments_dir.iterdir()):
            json_path = entry / "experiment.json"
            if not json_path.is_file():
                continue
            items.append(Experiment.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
        return items

    def list_archive_items(self) -> list[ArchivedResearchItem]:
        if not self.archive_dir.exists():
            return []
        items: list[ArchivedResearchItem] = []
        for entry in sorted(self.archive_dir.iterdir()):
            json_path = entry / "archive.json"
            if not json_path.is_file():
                continue
            items.append(
                ArchivedResearchItem.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            )
        return items

    def resurrect_archive_item(self, archive_id: str) -> Path:
        self.ensure_layout()
        entry = self.load_archive_item(archive_id)
        archive_root = self.archive_dir / archive_id
        artifact_root = archive_root / "artifact"
        destination = self.root / entry.original_relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        shutil.move(str(artifact_root), str(destination))
        shutil.rmtree(archive_root)
        return destination

    def _source_path_for_kind(self, kind: str, item_id: str) -> Path:
        mapping = {
            "theme": self.themes_dir / item_id,
            "unknown": self.unknowns_dir / item_id,
            "rabbit_hole": self.rabbit_holes_dir / item_id,
            "publication": self.publications_dir / item_id,
            "citation": self.citations_dir / item_id,
            "inspiration": self.inspiration_dir / item_id,
            "section_map": self.section_maps_dir / item_id,
            "observation": self.observations_dir / item_id,
            "opportunity": self.opportunities_dir / item_id,
            "hypothesis": self.hypotheses_dir / item_id,
            "experiment_plan": self.experiment_plans_dir / item_id,
            "promotion_candidate": self.promotion_candidates_dir / item_id,
            "experiment": self.experiments_dir / item_id,
        }
        try:
            return mapping[kind]
        except KeyError as exc:
            raise ValueError(f"unsupported archive kind: {kind!r}") from exc

    def search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (
            "vision",
            "theme",
            "unknown",
            "rabbit_hole",
            "publication",
            "section_map",
            "observation",
            "opportunity",
            "hypothesis",
            "experiment_plan",
        ),
        limit: int = 10,
    ) -> list[ResearchSearchHit]:
        needle = query.lower().strip()
        if not needle:
            return []
        hits: list[ResearchSearchHit] = []
        if "rabbit_hole" in kinds and self.rabbit_holes_dir.exists():
            for entry in sorted(self.rabbit_holes_dir.iterdir()):
                json_path = entry / "rabbit_hole.json"
                if not json_path.is_file():
                    continue
                rabbit_hole = RabbitHole.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
                haystack = " ".join(
                    (
                        rabbit_hole.title,
                        rabbit_hole.question,
                        rabbit_hole.scope,
                        rabbit_hole.theme,
                        " ".join(rabbit_hole.related_sections),
                        " ".join(rabbit_hole.tags),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="rabbit_hole",
                            id=rabbit_hole.id,
                            title=rabbit_hole.title,
                            summary=rabbit_hole.question,
                            path=entry,
                        )
                    )
        if "vision" in kinds and (self.vision_dir / "vision.json").is_file():
            vision = self.load_vision()
            haystack = " ".join(
                (
                    vision.title,
                    vision.summary,
                    " ".join(vision.themes),
                    " ".join(vision.success_metrics),
                )
            ).lower()
            if needle in haystack:
                hits.append(
                    ResearchSearchHit(
                        kind="vision",
                        id=vision.id,
                        title=vision.title,
                        summary=vision.summary,
                        path=self.vision_dir,
                    )
                )
        if "theme" in kinds and self.themes_dir.exists():
            for entry in sorted(self.themes_dir.iterdir()):
                json_path = entry / "theme.json"
                if not json_path.is_file():
                    continue
                theme = Theme.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (theme.title, theme.description, theme.priority, theme.status, theme.vision_id)
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="theme",
                            id=theme.id,
                            title=theme.title,
                            summary=theme.description,
                            path=entry,
                        )
                    )
        if "unknown" in kinds and self.unknowns_dir.exists():
            for entry in sorted(self.unknowns_dir.iterdir()):
                json_path = entry / "unknown.json"
                if not json_path.is_file():
                    continue
                unknown = Unknown.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (
                        unknown.question,
                        unknown.why_it_matters,
                        unknown.current_belief,
                        unknown.status,
                        unknown.theme_id,
                        " ".join(unknown.related_sections),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="unknown",
                            id=unknown.id,
                            title=unknown.question,
                            summary=unknown.why_it_matters,
                            path=entry,
                        )
                    )
        if "publication" in kinds and self.publications_dir.exists():
            for entry in sorted(self.publications_dir.iterdir()):
                json_path = entry / "publication.json"
                if not json_path.is_file():
                    continue
                publication = Publication.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
                haystack = " ".join(
                    (
                        publication.title,
                        publication.summary,
                        " ".join(publication.claims),
                        " ".join(publication.recommendations),
                        " ".join(publication.open_questions),
                        " ".join(publication.sources),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="publication",
                            id=publication.id,
                            title=publication.title,
                            summary=publication.summary,
                            path=entry,
                        )
                    )
        if "citation" in kinds and self.citations_dir.exists():
            for entry in sorted(self.citations_dir.iterdir()):
                json_path = entry / "citation.json"
                if not json_path.is_file():
                    continue
                citation = Citation.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (
                        citation.source_publication_id,
                        citation.target_publication_id,
                        citation.relationship,
                        citation.note,
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="citation",
                            id=citation.id,
                            title=f"{citation.source_publication_id} -> {citation.target_publication_id}",
                            summary=citation.relationship,
                            path=entry,
                        )
                    )
        if "inspiration" in kinds and self.inspiration_dir.exists():
            for entry in sorted(self.inspiration_dir.iterdir()):
                json_path = entry / "inspiration.json"
                if not json_path.is_file():
                    continue
                note = InspirationNote.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (
                        note.title,
                        note.summary,
                        note.source.kind,
                        note.source.ref,
                        note.source.title,
                        note.source.excerpt,
                        " ".join(note.related_themes),
                        " ".join(note.related_sections),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="inspiration",
                            id=note.id,
                            title=note.title,
                            summary=note.summary,
                            path=entry,
                        )
                    )
        if "section_map" in kinds and self.section_maps_dir.exists():
            for entry in sorted(self.section_maps_dir.iterdir()):
                json_path = entry / "section_map.json"
                if not json_path.is_file():
                    continue
                section_map = SectionMap.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
                haystack = " ".join(
                    (
                        section_map.section,
                        " ".join(section_map.files),
                        " ".join(section_map.interfaces),
                        " ".join(section_map.constraints),
                        " ".join(section_map.weaknesses),
                        " ".join(section_map.opportunities),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="section_map",
                            id=section_map.id,
                            title=section_map.section,
                            summary="; ".join(section_map.opportunities or section_map.weaknesses),
                            path=entry,
                        )
                    )
        if "observation" in kinds and self.observations_dir.exists():
            for entry in sorted(self.observations_dir.iterdir()):
                json_path = entry / "observation.json"
                if not json_path.is_file():
                    continue
                observation = Observation.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
                haystack = " ".join(
                    (
                        observation.title,
                        observation.summary,
                        observation.source_type,
                        observation.source_ref,
                        " ".join(observation.related_sections),
                        observation.theme,
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="observation",
                            id=observation.id,
                            title=observation.title,
                            summary=observation.summary,
                            path=entry,
                        )
                    )
        if "opportunity" in kinds and self.opportunities_dir.exists():
            for entry in sorted(self.opportunities_dir.iterdir()):
                json_path = entry / "opportunity.json"
                if not json_path.is_file():
                    continue
                opportunity = Opportunity.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
                haystack = " ".join(
                    (
                        opportunity.title,
                        opportunity.summary,
                        " ".join(opportunity.related_sections),
                        " ".join(opportunity.origin_observations),
                        " ".join(opportunity.change_modes),
                        opportunity.theme,
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="opportunity",
                            id=opportunity.id,
                            title=opportunity.title,
                            summary=opportunity.summary,
                            path=entry,
                        )
                    )
        if "hypothesis" in kinds and self.hypotheses_dir.exists():
            for entry in sorted(self.hypotheses_dir.iterdir()):
                json_path = entry / "hypothesis.json"
                if not json_path.is_file():
                    continue
                hypothesis = Hypothesis.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (
                        hypothesis.claim,
                        hypothesis.expected_win,
                        hypothesis.risk_level,
                        hypothesis.change_mode,
                        hypothesis.opportunity_id,
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="hypothesis",
                            id=hypothesis.id,
                            title=hypothesis.claim,
                            summary=hypothesis.expected_win,
                            path=entry,
                        )
                    )
        if "experiment_plan" in kinds and self.experiment_plans_dir.exists():
            for entry in sorted(self.experiment_plans_dir.iterdir()):
                json_path = entry / "experiment_plan.json"
                if not json_path.is_file():
                    continue
                plan = ExperimentPlan.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                haystack = " ".join(
                    (
                        plan.plan,
                        plan.hypothesis_id,
                        " ".join(plan.target_files),
                        " ".join(plan.checks),
                        " ".join(plan.eval_slices),
                    )
                ).lower()
                if needle in haystack:
                    hits.append(
                        ResearchSearchHit(
                            kind="experiment_plan",
                            id=plan.id,
                            title=plan.hypothesis_id,
                            summary=plan.plan,
                            path=entry,
                        )
                    )
        return hits[:limit]

    @staticmethod
    def parse_change_intent(
        *,
        mode: str | None,
        subsystem: str | None,
        rationale: str | None,
        expected_outcome: str | None,
        risk: str | None,
    ) -> ChangeIntent | None:
        if not any((mode, subsystem, rationale, expected_outcome, risk)):
            return None
        if not all((mode, subsystem, rationale, expected_outcome)):
            raise ValueError(
                "change intent requires mode, subsystem, rationale, and expected_outcome together"
            )
        return ChangeIntent(
            mode=str(mode),  # type: ignore[arg-type]
            subsystem=str(subsystem).strip(),
            rationale=str(rationale).strip(),
            expected_outcome=str(expected_outcome).strip(),
            risk=str(risk or "medium").strip() or "medium",
        )


__all__ = [
    "ResearchSearchHit",
    "ResearchStore",
    "_split_csv",
    "default_research_root",
]
