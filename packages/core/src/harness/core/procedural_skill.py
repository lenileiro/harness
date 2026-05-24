"""Compatibility surface for the L2 procedural skill / tips subsystem.

The implementation is split into focused modules:

- `tips_models.py`: tip records, file-backed library, path helpers
- `tips_providers.py`: provider protocols and provider composition
- `tips_mining.py`: offline mining prompt/render/parse helpers

This module preserves the original import surface for runtime code and tests.
"""

from __future__ import annotations

from harness.core.tips_mining import MiningInput, parse_mined_tips, render_mining_prompt
from harness.core.tips_models import (
    Tip,
    TipLibrary,
    default_experience_paths,
    default_tip_paths,
    keywords_from_text,
)
from harness.core.tips_providers import (
    ArtifactTipProvider,
    CompositeTipsProvider,
    StaticTipsProvider,
    TipsProvider,
)

__all__ = [
    "ArtifactTipProvider",
    "CompositeTipsProvider",
    "MiningInput",
    "StaticTipsProvider",
    "Tip",
    "TipLibrary",
    "TipsProvider",
    "default_experience_paths",
    "default_tip_paths",
    "keywords_from_text",
    "parse_mined_tips",
    "render_mining_prompt",
]
