"""Versioned offline trajectories and deterministic training-data preparation.

These functions never execute tools, fetch attachments, download tokenizers, or
start training. Imports remain dataset records, not resumable agent sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import Message, ToolCall

DatasetFormat = Literal["harness", "openai", "trl", "sharegpt"]


class TrajectoryError(ValueError):
    pass


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    format: Literal["harness.trajectory"] = "harness.trajectory"
    version: Literal[1] = 1
    session: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] = Field(min_length=1, max_length=10000)
    tools: list[dict[str, Any]] = Field(default_factory=list, max_length=1000)
    runs: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (ValueError, TypeError, RecursionError):
        raise TrajectoryError("trajectory must contain finite JSON values") from None


def validate_messages(messages: list[Message], *, require_complete: bool = False) -> None:
    """Reject orphan/duplicate/interleaved results; unfinished tail calls may archive."""
    pending: dict[str, str] = {}
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message.role != "tool" and pending:
            raise TrajectoryError(
                f"message {index}: tool results must follow their assistant call before another turn"
            )
        if message.role in {"system", "user"}:
            if message.tool_calls or message.tool_call_id or message.name:
                raise TrajectoryError(f"message {index}: user/system message has tool-only fields")
            if message.content is None and not message.attachments:
                raise TrajectoryError(f"message {index}: content or attachments are required")
        elif message.role == "assistant":
            if message.tool_call_id or message.name:
                raise TrajectoryError(f"message {index}: assistant has result-only fields")
            if not message.content and not message.tool_calls and not message.attachments:
                raise TrajectoryError(f"message {index}: assistant has no content or action")
            for call in message.tool_calls or []:
                if not call.id or not call.name or call.id in seen:
                    raise TrajectoryError(
                        f"message {index}: tool IDs and names must be nonempty and IDs unique"
                    )
                seen.add(call.id)
                pending[call.id] = call.name
        else:
            if (
                message.tool_calls
                or not message.tool_call_id
                or pending.get(message.tool_call_id) != message.name
            ):
                raise TrajectoryError(
                    f"message {index}: orphan, duplicate or mismatched tool result"
                )
            if message.content is None and not message.attachments:
                raise TrajectoryError(f"message {index}: tool result content is required")
            del pending[message.tool_call_id]
    if require_complete and pending:
        raise TrajectoryError(
            "trajectory ends with unresolved tool calls; archive is valid but training/compression requires completed pairs"
        )


def validate_trajectory(trajectory: Trajectory, *, require_complete: bool = False) -> None:
    _json(trajectory.model_dump())
    validate_messages(trajectory.messages, require_complete=require_complete)
    names: set[str] = set()
    for tool in trajectory.tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise TrajectoryError("tools must use public function JSON schemas")
        name = function.get("name")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(parameters, dict)
            or parameters.get("type") != "object"
        ):
            raise TrajectoryError("tool schemas require unique names and object parameters")
        names.add(name)


def _external_messages(raw: list[Any], *, sharegpt: bool) -> list[Message]:
    messages: list[Message] = []
    pending: dict[str, str] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TrajectoryError(f"message {index} must be an object")
        allowed = (
            {"from", "value", "tool_calls", "tool_call_id", "name"}
            if sharegpt
            else {"role", "content", "tool_calls", "tool_call_id", "name", "attachments"}
        )
        if set(item) - allowed:
            raise TrajectoryError(
                f"message {index} has unsupported fields; convert them explicitly instead of losing data"
            )
        role = item.get("from") if sharegpt else item.get("role")
        if not isinstance(role, str):
            raise TrajectoryError(f"message {index}: role is required")
        if sharegpt:
            role = {
                "human": "user",
                "gpt": "assistant",
                "system": "system",
                "tool": "tool",
                "function": "tool",
            }.get(role, role)
        calls: list[ToolCall] = []
        for number, raw_call in enumerate(item.get("tool_calls") or []):
            if not isinstance(raw_call, dict) or raw_call.get("type", "function") != "function":
                raise TrajectoryError(f"message {index}: unsupported tool call")
            function = raw_call.get("function")
            if (
                not isinstance(function, dict)
                or set(raw_call) - {"id", "type", "function"}
                or set(function) - {"name", "arguments"}
            ):
                raise TrajectoryError(f"message {index}: unsupported function call fields")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, RecursionError):
                    raise TrajectoryError(
                        f"message {index}: malformed tool argument JSON"
                    ) from None
            call = ToolCall.model_validate(
                {
                    "id": raw_call.get("id", f"import_{index}_{number}"),
                    "name": function.get("name"),
                    "arguments": arguments,
                }
            )
            calls.append(call)
            pending[call.id] = call.name
        call_id = item.get("tool_call_id")
        name = item.get("name")
        if role == "tool":
            if call_id is None:
                matches = [key for key, value in pending.items() if name is None or value == name]
                if len(matches) != 1:
                    raise TrajectoryError(
                        f"message {index}: tool result needs an explicit ID for ambiguous parallel calls"
                    )
                call_id = matches[0]
            name = name or pending.get(call_id)
            pending.pop(call_id, None)
        message = Message.model_validate(
            {
                "role": role,
                "content": item.get("value") if sharegpt else item.get("content"),
                "tool_calls": calls or None,
                "tool_call_id": call_id,
                "name": name,
                "attachments": item.get("attachments", []),
            }
        )
        messages.append(message)
    return messages


def import_record(
    record: dict[str, Any],
    *,
    source_format: DatasetFormat = "harness",
    tools: list[dict[str, Any]] | None = None,
) -> Trajectory:
    try:
        _json(record)
        if source_format == "harness":
            trajectory = Trajectory.model_validate(record)
            if tools is not None:
                if trajectory.tools and trajectory.tools != tools:
                    raise TrajectoryError("provided schemas conflict with the trajectory's schemas")
                trajectory.tools = tools
        else:
            key = "conversations" if source_format == "sharegpt" else "messages"
            if not isinstance(record.get(key), list) or set(record) - {key, "tools", "metadata"}:
                raise TrajectoryError(
                    "dataset row has unsupported columns or lacks its message list"
                )
            declared = record.get("tools", tools or [])
            if tools is not None and record.get("tools") and record["tools"] != tools:
                raise TrajectoryError("provided schemas conflict with dataset schemas")
            trajectory = Trajectory(
                messages=_external_messages(record[key], sharegpt=source_format == "sharegpt"),
                tools=declared,
                metadata={
                    "import_format": source_format,
                    "source_metadata": record.get("metadata", {}),
                },
            )
        validate_trajectory(trajectory)
        return trajectory
    except TrajectoryError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise TrajectoryError(
            "trajectory has invalid fields; check schema, roles, tool arguments and attachments"
        ) from None


def parse_jsonl(
    data: bytes,
    *,
    source_format: DatasetFormat = "harness",
    tools: list[dict[str, Any]] | None = None,
    max_records: int = 10000,
) -> list[Trajectory]:
    result: list[Trajectory] = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > 24 * 1024 * 1024 or len(result) >= max_records:
            raise TrajectoryError("dataset exceeds the record size or count limit")
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TrajectoryError("record must be a JSON object")
            result.append(import_record(record, source_format=source_format, tools=tools))
        except (ValueError, RecursionError) as exc:
            detail = str(exc) if isinstance(exc, TrajectoryError) else "invalid JSON"
            raise TrajectoryError(f"line {index}: {detail}") from None
    if not result:
        raise TrajectoryError("dataset contains no records")
    return result


def content_fingerprint(trajectory: Trajectory) -> str:
    # Identity/provenance does not turn duplicate conversations into independent
    # training examples or permit them to leak across train/evaluation splits.
    messages = [
        message.model_dump(mode="json", exclude={"cache_breakpoint"})
        for message in trajectory.messages
    ]
    calls = [call for message in messages for call in message.get("tool_calls") or []]
    identifiers = {call["id"]: f"call_{index}" for index, call in enumerate(calls)}
    for message in messages:
        for call in message.get("tool_calls") or []:
            call["id"] = identifiers[call["id"]]
        if message.get("tool_call_id") in identifiers:
            message["tool_call_id"] = identifiers[message["tool_call_id"]]
    payload = {"messages": messages, "tools": trajectory.tools}
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def estimated_tokens(trajectory: Trajectory) -> int:
    """Deterministic byte estimate including schemas/media, not a model tokenizer."""
    payload = {
        "messages": [
            message.model_dump(mode="json", exclude_defaults=True)
            for message in trajectory.messages
        ],
        "tools": trajectory.tools,
    }
    return math.ceil(len(_json(payload).encode()) / 4)


def compress_trajectory(
    trajectory: Trajectory, *, max_tokens: int, tool_result_chars: int = 2000
) -> Trajectory:
    if max_tokens < 1 or tool_result_chars < 64:
        raise TrajectoryError(
            "compression budget must be positive and tool result limit at least 64 characters"
        )
    validate_trajectory(trajectory, require_complete=True)
    output = trajectory.model_copy(deep=True)
    before = estimated_tokens(output)
    truncated: list[str] = []
    removed: list[int] = []
    if before > max_tokens:
        for message in output.messages:
            if (
                message.role == "tool"
                and message.content
                and len(message.content) > tool_result_chars
            ):
                half = (tool_result_chars - 48) // 2
                message.content = (
                    message.content[:half]
                    + "\n[tool output shortened for dataset]\n"
                    + message.content[-half:]
                )
                truncated.append(message.tool_call_id or "")
    # Keep first request, all system instructions, last user request and final
    # assistant answer. Assistant call plus all results is always one unit.
    blocks: list[list[int]] = []
    index = 0
    while index < len(output.messages):
        start = index
        message = output.messages[index]
        index += 1
        if message.tool_calls:
            index += len(message.tool_calls)
        blocks.append(list(range(start, index)))
    users = [i for i, message in enumerate(output.messages) if message.role == "user"]
    protected = {i for i, message in enumerate(output.messages) if message.role == "system"}
    protected.update(users[:1] + users[-1:] + [len(output.messages) - 1])
    original = output.messages
    for block in blocks:
        if estimated_tokens(output) <= max_tokens:
            break
        if protected.intersection(block):
            continue
        removed.extend(block)
        output.messages = [message for i, message in enumerate(original) if i not in set(removed)]
    if estimated_tokens(output) > max_tokens:
        raise TrajectoryError(
            "protected instructions, requests, answer, schemas or attachments exceed the budget; increase it or review the source manually"
        )
    output.metadata["compression"] = {
        "version": 1,
        "source_sha256": content_fingerprint(trajectory),
        "estimator": "utf8-json-bytes/4",
        "before_estimated_tokens": before,
        "after_estimated_tokens": estimated_tokens(output),
        "removed_message_indices": removed,
        "shortened_tool_call_ids": truncated,
    }
    validate_trajectory(output, require_complete=True)
    return output


def training_record(
    trajectory: Trajectory, *, target: Literal["openai", "trl", "sharegpt"] = "trl"
) -> dict[str, Any]:
    validate_trajectory(trajectory, require_complete=True)
    if trajectory.messages[-1].role != "assistant" or trajectory.messages[-1].tool_calls:
        raise TrajectoryError("training records must end with a completed assistant answer")
    if not any(message.role == "user" for message in trajectory.messages):
        raise TrajectoryError("training records require a user request")
    if any(message.attachments for message in trajectory.messages):
        raise TrajectoryError(
            "text training adapters cannot silently discard media; use the versioned archive or a media-aware trainer"
        )
    names = {tool["function"]["name"] for tool in trajectory.tools}
    called = {call.name for message in trajectory.messages for call in message.tool_calls or []}
    if called - names:
        raise TrajectoryError(
            "tool-call training requires the original function schemas; provide --tools, never infer schemas from one call"
        )
    result: list[dict[str, Any]] = []
    for message in trajectory.messages:
        item: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _json(call.arguments)
                        if target == "openai"
                        else call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == "tool":
            item.update(tool_call_id=message.tool_call_id, name=message.name)
        if target == "sharegpt":
            item["from"] = {"user": "human", "assistant": "gpt"}.get(item.pop("role"), message.role)
            if "content" in item:
                item["value"] = item.pop("content")
        result.append(item)
    record: dict[str, Any] = {"conversations" if target == "sharegpt" else "messages": result}
    if trajectory.tools:
        record["tools"] = trajectory.tools
    return record


def prepare_dataset(
    trajectories: list[Trajectory],
    *,
    target: Literal["openai", "trl", "sharegpt"] = "trl",
    seed: int = 42,
    sample: int | None = None,
    eval_fraction: float = 0.1,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 <= eval_fraction < 1 or (sample is not None and sample < 1):
        raise TrajectoryError("evaluation fraction must be in [0,1) and sample size positive")
    if not trajectories:
        raise TrajectoryError("dataset contains no records")
    unique: dict[str, Trajectory] = {}
    for trajectory in trajectories:
        derived = (
            compress_trajectory(trajectory, max_tokens=max_tokens) if max_tokens else trajectory
        )
        fingerprint = content_fingerprint(derived)
        previous = unique.get(fingerprint)
        if previous is None or _json(derived.model_dump(mode="json")) < _json(
            previous.model_dump(mode="json")
        ):
            unique[fingerprint] = derived
    ordered = sorted(unique.items())
    random.Random(seed).shuffle(ordered)
    if sample is not None:
        if sample > len(ordered):
            raise TrajectoryError("requested sample exceeds the unique record count")
        ordered = ordered[:sample]
    evaluation_count = (
        min(len(ordered) - 1, math.ceil(len(ordered) * eval_fraction)) if ordered else 0
    )
    evaluation = ordered[:evaluation_count]
    train = ordered[evaluation_count:]
    train_rows = [training_record(record, target=target) for _, record in train]
    eval_rows = [training_record(record, target=target) for _, record in evaluation]
    report = {
        "format": "harness.dataset-manifest",
        "version": 1,
        "target": target,
        "seed": seed,
        "input_records": len(trajectories),
        "unique_records": len(unique),
        "training_records": len(train),
        "evaluation_records": len(evaluation),
        "training_sha256": [key for key, _ in train],
        "evaluation_sha256": [key for key, _ in evaluation],
        "token_estimator": "utf8-json-bytes/4",
        "max_estimated_tokens": max_tokens,
        "compression": {
            key: record.metadata["compression"]
            for key, record in ordered
            if "compression" in record.metadata
        },
    }
    return train_rows, eval_rows, report
