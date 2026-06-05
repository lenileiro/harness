from __future__ import annotations

import re
import unicodedata


def slugify(value: str, *, fallback: str = "untitled") -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug or fallback
