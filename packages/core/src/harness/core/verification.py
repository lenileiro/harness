"""Post-run verification: did the agent actually accomplish the user's goal?

A `Verifier` looks at the completed session and its activity ledger, and
emits a `VerificationResult` saying whether the run can finish or needs
human review / a follow-up.

The runtime calls the configured verifier once, after the terminal `Done`
event, and yields a `Verification(result=...)` event so consumers can
render the verdict. It also records a `verification.completed` activity
entry so the verdict is part of the durable ledger.

Two real verifiers ship in core:

- `RuleVerifier`: pure-Python; checks the activity ledger for tool errors.
- `LLMJudgeVerifier`: one extra LLM call against any `Adapter` to judge
  whether the assistant's final answer addresses the user's goal.

Both are wired and tested. There are no schema-only verifiers — every
Verifier in this module has a real implementation behind it.
"""

from __future__ import annotations

from harness.core import verification_behavioral as _behavioral
from harness.core import verification_guards as _guards
from harness.core import verification_judges as _judges
from harness.core import verification_structural as _structural
from harness.core.schemas import VerificationResult

_looks_like_feature_add = _structural.looks_like_feature_add
_first_user_prompt = _structural.first_user_prompt
asyncio = _judges.asyncio


Verifier = _judges.Verifier
RuleVerifier = _judges.RuleVerifier
LLMJudgeVerifier = _judges.LLMJudgeVerifier
VerifierRouter = _judges.VerifierRouter
WorkItemJudge = _judges.WorkItemJudge
_is_repetitive = _judges._is_repetitive

EvidenceCheckKind = _guards.EvidenceCheckKind
EvidenceContract = _guards.EvidenceContract
EvidenceContractResult = _guards.EvidenceContractResult
evaluate_evidence = _guards.evaluate_evidence
VerificationGateway = _guards.VerificationGateway
ClaimGroundingVerifier = _guards.ClaimGroundingVerifier
StateVerifier = _guards.StateVerifier
ConsensusVerifier = _guards.ConsensusVerifier

ChainedVerifier = _structural.ChainedVerifier
ShellVerifier = _structural.ShellVerifier
VerifyBeforeDoneVerifier = _structural.VerifyBeforeDoneVerifier
MinimalFixVerifier = _structural.MinimalFixVerifier
PhaseGateVerifier = _structural.PhaseGateVerifier
TestsBeforeEditVerifier = _structural.TestsBeforeEditVerifier
FileScopeVerifier = _structural.FileScopeVerifier

MisdirectedSuggestionVerifier = _behavioral.MisdirectedSuggestionVerifier
PromptSurfaceRevertVerifier = _behavioral.PromptSurfaceRevertVerifier
NegativeConstraintVerifier = _behavioral.NegativeConstraintVerifier
BugfixCommentRewriteVerifier = _behavioral.BugfixCommentRewriteVerifier
DiagnosisAlignmentVerifier = _behavioral.DiagnosisAlignmentVerifier


__all__ = [
    "ChainedVerifier",
    "ClaimGroundingVerifier",
    "ConsensusVerifier",
    "DiagnosisAlignmentVerifier",
    "EvidenceCheckKind",
    "EvidenceContract",
    "EvidenceContractResult",
    "FileScopeVerifier",
    "LLMJudgeVerifier",
    "MinimalFixVerifier",
    "MisdirectedSuggestionVerifier",
    "NegativeConstraintVerifier",
    "PhaseGateVerifier",
    "PromptSurfaceRevertVerifier",
    "RuleVerifier",
    "ShellVerifier",
    "StateVerifier",
    "TestsBeforeEditVerifier",
    "VerificationGateway",
    "VerificationResult",
    "Verifier",
    "VerifierRouter",
    "VerifyBeforeDoneVerifier",
    "WorkItemJudge",
    "_is_repetitive",
    "evaluate_evidence",
]
