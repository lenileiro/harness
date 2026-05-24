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
    default_tip_paths,
    keywords_from_text,
)
from harness.core.tips_providers import (
    ArtifactTipProvider,
    CompositeTipsProvider,
    StaticTipsProvider,
    TipsProvider,
    default_experience_roots,
    load_default_experience_provider,
)

__all__ = [
    "ArtifactTipProvider",
    "CompositeTipsProvider",
    "MiningInput",
    "StaticTipsProvider",
    "Tip",
    "TipLibrary",
    "TipsProvider",
    "default_experience_roots",
    "default_tip_paths",
    "keywords_from_text",
    "load_default_experience_provider",
    "parse_mined_tips",
    "render_mining_prompt",
]
