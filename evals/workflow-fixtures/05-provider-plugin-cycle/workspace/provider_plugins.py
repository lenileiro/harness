from harness.core.domain_profiles import DomainProfile
from harness.core.schemas import VerificationResult
from harness.core.tips_models import Tip


class RepoHintsProvider:
    def query(self, task_text: str, *, top_k: int = 3):
        return [
            Tip(
                text=f"Stay inside the repo for: {task_text}",
                triggers=("repo", "local"),
                weight=1.0,
            )
        ][:top_k]


class DocsReviewProfileProvider:
    def profiles(self):
        return [DomainProfile(name="docs-review-pack", description="Docs-only review pack")]


class _StrictVerifier:
    async def verify(self, *, session, activity):
        return VerificationResult(can_finish=True, confidence=1.0, reason="ok")


class StrictVerifierProvider:
    def verifiers(self):
        return [_StrictVerifier()]


class _SkepticCritic:
    async def critique(self, *, session, verification_result, activity):
        return "Challenge the broad assumption."


class SkepticCriticProvider:
    def critics(self):
        return [_SkepticCritic()]
