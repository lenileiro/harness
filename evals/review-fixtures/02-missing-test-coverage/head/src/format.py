def normalize_email(value: str) -> str:
    cleaned = value.strip()
    if cleaned == "":
        return "(missing)"
    return cleaned.lower()
