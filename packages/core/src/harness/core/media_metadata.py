"""Read duration from already bounded audio bytes without decoding or fetching URLs."""

from __future__ import annotations

import io
import math

from tinytag import TinyTag, TinyTagException


def audio_duration_ms(raw: bytes) -> int | None:
    if not raw or len(raw) > 20 * 1024 * 1024:
        return None
    try:
        duration = TinyTag.get(file_obj=io.BytesIO(raw), tags=False, image=False).duration
    except (TinyTagException, OSError, ValueError):
        return None
    if duration is None or not math.isfinite(duration) or not 0 < duration <= 86_400:
        return None
    return math.ceil(duration * 1000)
