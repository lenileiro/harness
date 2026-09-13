from __future__ import annotations

import pytest

from harness.core.schemas import MediaAttachment, Message, ToolCall
from harness.core.trajectories import (
    Trajectory,
    TrajectoryError,
    compress_trajectory,
    content_fingerprint,
    estimated_tokens,
    import_record,
    parse_jsonl,
    prepare_dataset,
    training_record,
    validate_trajectory,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def example(*, question="Read the files", large=False):
    return Trajectory(
        session={
            "id": "example",
            "metadata": {"memory_scope": {"workspace": "/example", "user_id": "owner"}},
        },
        tools=TOOLS,
        messages=[
            Message(role="system", content="Use tools carefully."),
            Message(role="user", content=question),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="first", name="read", arguments={"path": "a"}),
                    ToolCall(id="second", name="read", arguments={"path": "b"}),
                ],
            ),
            Message(
                role="tool",
                name="read",
                tool_call_id="second",
                content="B" * (10000 if large else 1),
            ),
            Message(
                role="tool",
                name="read",
                tool_call_id="first",
                content="A" * (10000 if large else 1),
            ),
            Message(role="assistant", content="Both files were read."),
        ],
    )


@pytest.mark.parametrize("target", ["openai", "trl", "sharegpt"])
def test_public_formats_round_trip_exact_parallel_tool_relationships(target):
    source = example()
    row = training_record(source, target=target)
    result = import_record(row, source_format=target)
    assert result.messages == source.messages and result.tools == source.tools
    if target == "openai":
        assert isinstance(row["messages"][2]["tool_calls"][0]["function"]["arguments"], str)
    if target == "trl":
        assert isinstance(row["messages"][2]["tool_calls"][0]["function"]["arguments"], dict)


@pytest.mark.parametrize(
    "change", ["orphan", "duplicate", "wrong-name", "interleaved", "duplicate-call"]
)
def test_validation_rejects_broken_protocol(change):
    record = example()
    if change == "orphan":
        record.messages[3].tool_call_id = "not-called"
    elif change == "duplicate":
        record.messages[4] = record.messages[3].model_copy()
    elif change == "wrong-name":
        record.messages[3].name = "shell"
    elif change == "interleaved":
        record.messages.insert(3, Message(role="user", content="interrupt"))
    else:
        assert record.messages[2].tool_calls is not None
        record.messages[2].tool_calls[1].id = "first"
    with pytest.raises(TrajectoryError):
        validate_trajectory(record)


def test_incomplete_tail_can_be_archived_but_not_trained_or_compressed():
    record = example()
    record.messages = record.messages[:3]
    validate_trajectory(record)
    assert parse_jsonl((record.model_dump_json() + "\n").encode())[0] == record
    with pytest.raises(TrajectoryError, match="unresolved"):
        training_record(record)
    with pytest.raises(TrajectoryError, match="unresolved"):
        compress_trajectory(record, max_tokens=1000)


def test_compression_preserves_complete_pairs_and_reports_omissions_without_mutation():
    record = example(large=True)
    saved = record.model_dump_json()
    compressed = compress_trajectory(record, max_tokens=500, tool_result_chars=128)
    assert estimated_tokens(compressed) <= 500
    assert compressed.messages[0] == record.messages[0]
    assert compressed.messages[1] == record.messages[1]
    assert compressed.messages[-1] == record.messages[-1]
    assert set(compressed.metadata["compression"]["shortened_tool_call_ids"]) == {"first", "second"}
    validate_trajectory(compressed, require_complete=True)
    assert record.model_dump_json() == saved
    smaller = compress_trajectory(
        record,
        max_tokens=estimated_tokens(
            Trajectory(
                messages=[record.messages[0], record.messages[1], record.messages[-1]], tools=TOOLS
            )
        ),
        tool_result_chars=128,
    )
    assert smaller.metadata["compression"]["removed_message_indices"] == [2, 3, 4]
    assert [message.role for message in smaller.messages] == ["system", "user", "assistant"]
    with pytest.raises(TrajectoryError, match="protected"):
        compress_trajectory(record, max_tokens=1)


def test_training_requires_original_schemas_and_refuses_silent_media_loss():
    record = example()
    record.tools = []
    with pytest.raises(TrajectoryError, match="schemas"):
        training_record(record)
    record = example()
    record.messages[1].attachments = [
        MediaAttachment(kind="image", mime_type="image/png", data="eA==")
    ]
    restored = parse_jsonl(record.model_dump_json().encode())[0]
    assert restored.messages[1].attachments == record.messages[1].attachments
    with pytest.raises(TrajectoryError, match="media"):
        training_record(restored)


def test_external_implicit_tool_ids_only_when_unambiguous():
    row = {
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "read", "arguments": {"path": "a"}}}
                ],
            },
            {"role": "tool", "name": "read", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": TOOLS,
    }
    result = import_record(row, source_format="trl")
    assert result.messages[1].tool_calls is not None
    assert result.messages[2].tool_call_id == result.messages[1].tool_calls[0].id
    row["messages"][1]["tool_calls"].append(
        {"type": "function", "function": {"name": "read", "arguments": {"path": "b"}}}
    )
    with pytest.raises(TrajectoryError, match="ambiguous"):
        import_record(row, source_format="trl")


def test_sampling_and_splits_are_reproducible_order_independent_and_deduplicated():
    records = [example(question=f"Question {number}") for number in range(10)]
    duplicate = records[0].model_copy(deep=True)
    duplicate.session = {"id": "different-session-same-data"}
    first = prepare_dataset([*records, duplicate], seed=7, sample=8, eval_fraction=0.25)
    second = prepare_dataset([*reversed(records), duplicate], seed=7, sample=8, eval_fraction=0.25)
    assert first == second
    train, evaluation, report = first
    assert len(train) == 6 and len(evaluation) == 2
    assert report["unique_records"] == 10
    assert set(report["training_sha256"]).isdisjoint(report["evaluation_sha256"])
    with pytest.raises(TrajectoryError, match="sample"):
        prepare_dataset(records, sample=11)
    assert content_fingerprint(duplicate) == content_fingerprint(records[0])


@pytest.mark.parametrize(
    "data",
    [
        b'{"format":"harness.trajectory","version":99,"messages":[]}\n',
        b"not-json",
        b"[]",
        b"",
        b'{"messages": [NaN]}',
    ],
)
def test_bad_jsonl_is_actionable_without_echoing_values(data):
    with pytest.raises(TrajectoryError):
        parse_jsonl(data)


def test_external_unknown_fields_are_not_silently_discarded():
    with pytest.raises(TrajectoryError, match="unsupported"):
        import_record(
            {
                "messages": [
                    {"role": "user", "content": "private-value", "audio": {"data": "do-not-drop"}}
                ]
            },
            source_format="openai",
        )
    with pytest.raises(TrajectoryError, match="finite"):
        validate_trajectory(
            Trajectory(
                messages=[Message(role="user", content="hi")], metadata={"score": float("nan")}
            )
        )
    with pytest.raises(TrajectoryError, match="argument"):
        import_record(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "read", "arguments": '{"private-value":INVALID}'}}
                        ],
                    }
                ]
            },
            source_format="openai",
        )


def test_duplicate_calls_with_different_ids_do_not_cross_dataset_splits():
    first = example()
    second = example()
    assert second.messages[2].tool_calls is not None
    for call in second.messages[2].tool_calls:
        call.id = "other_" + call.id
    for message in second.messages:
        if message.tool_call_id:
            message.tool_call_id = "other_" + message.tool_call_id
    assert content_fingerprint(first) == content_fingerprint(second)
    prepared = prepare_dataset([first, second], eval_fraction=0.5)
    assert prepared == prepare_dataset([second, first], eval_fraction=0.5)
    assert prepared[2]["unique_records"] == 1 and not prepared[1]
