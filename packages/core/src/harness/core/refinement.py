from __future__ import annotations

from harness.core.promotion_candidates import PromotionCandidate


def summarize_refinement(candidate: PromotionCandidate) -> list[str]:
    lines = [candidate.title, candidate.summary]
    if candidate.expected_metric:
        lines.append(f"expected_metric: {candidate.expected_metric}")
    if candidate.validation_plan:
        lines.append(f"validation_plan: {candidate.validation_plan}")
    return lines


__all__ = ["summarize_refinement"]
