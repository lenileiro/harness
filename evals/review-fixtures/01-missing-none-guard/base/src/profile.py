def display_name(user: dict[str, str] | None) -> str:
    if user is None:
        return "anonymous"
    return user.get("name", "anonymous")
