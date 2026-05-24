from __future__ import annotations

import json
from dataclasses import dataclass

from harness.core.tips_models import Tip, logger


@dataclass
class MiningInput:
    session_id: str
    task_text: str
    failure_summary: str
    transcript_excerpt: str


def render_mining_prompt(item: MiningInput) -> str:
    return (
        "You are a harness-improvement extractor. Read the failed agent "
        "session below and extract 0 to 3 short, *procedural* tips that would "
        "have prevented the failure. Each tip should be a single imperative "
        "sentence under 20 words. Do not paraphrase the task — extract a "
        "*generalizable lesson* about how to interact with this codebase or "
        "tooling.\n"
        f"\n--- TASK ---\n{item.task_text.strip()}\n"
        f"\n--- FAILURE ---\n{item.failure_summary.strip()}\n"
        f"\n--- TRANSCRIPT (tail) ---\n{item.transcript_excerpt.strip()}\n"
        "\nReturn ONLY valid JSON in this shape:\n"
        '  {"tips": [{"text": "...", "triggers": ["keyword", "..."]}]}\n'
        'Return `{"tips": []}` when no useful lesson can be extracted.'
    )


def parse_mined_tips(response: str, *, source_session_id: str | None = None) -> list[Tip]:
    body = response.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("tips.mining_bad_json", preview=body[:200])
        return []
    raw = payload.get("tips") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[Tip] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or len(text) > 200:
            continue
        triggers = item.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        trigger_values = [str(trigger).strip() for trigger in triggers if str(trigger).strip()]
        out.append(
            Tip(
                text=text,
                triggers=tuple(trigger_values),
                source_session_id=source_session_id,
            )
        )
    return out


__all__ = [
    "MiningInput",
    "parse_mined_tips",
    "render_mining_prompt",
]
