"""Atomic clarification answers and A2A message receipts, separate from approvals."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from a2a.utils.errors import InvalidParamsError

from harness.core.clarification import (
    PendingQuestion,
    apply_question_answers,
    parse_question_answer,
)

if TYPE_CHECKING:
    from harness.server.service import HarnessService


async def pending_question(
    service: HarnessService, owner: str, session_id: str
) -> PendingQuestion | None:
    session = await service._owned_session(owner, session_id)
    if session is None or not session.metadata.get("pending_question_id"):
        return None
    record, _ = await service.questions._owned(owner, str(session.metadata["pending_question_id"]))
    return record


def question_detail(record: PendingQuestion) -> str:
    if record.status == "pending":
        lines = [f"Clarification required: {record.id}"]
        for index, question in enumerate(record.questions):
            if f"q{index}" in record.answers:
                continue
            lines.append(f"q{index}: {question.question}")
            for number, choice in enumerate(question.choices or [], 1):
                lines.append(f"  {number}. {choice}")
            if question.choices:
                lines.append("  Your own answer is also accepted.")
            if question.multi_select and question.choices:
                lines.append("  Multiple selections allowed: comma-separated text or a JSON list.")
        lines.extend(
            [
                f"Expires: {record.expires_at.isoformat(timespec='seconds')}",
                "Send a new messageId with this taskId and a text answer for the next question, "
                'or text containing a JSON object such as {"q0":"answer","q1":["choice"]}. '
                "Partial answers are saved; repeated message IDs never answer another question. "
                "Answers do not approve tool actions.",
            ]
        )
        return "\n".join(lines)
    return (
        f"Clarification {record.id} is {record.status}. Saved answers are retained. "
        "Send a new messageId with this taskId to continue. Outstanding tool approvals must still be resolved separately."
    )


async def find_receipt(
    service: HarnessService, owner: str, message_id: str, digest: str
) -> dict[str, Any] | None:
    rows = await service.store.rows(
        "SELECT * FROM api_a2a_answer_receipts WHERE owner=? AND message_id=?",
        (owner, message_id),
    )
    if not rows:
        return None
    if rows[0]["digest"] != digest:
        raise InvalidParamsError("messageId was already used for different content")
    return rows[0]


async def record_answer(
    service: HarnessService,
    owner: str,
    *,
    task_id: str,
    run_id: str,
    record: PendingQuestion,
    message_id: str,
    digest: str,
    text: str,
) -> dict[str, Any]:
    """Persist both the normalized qN selection and its message identity together.

    A complete answer's receipt stays resumable until the ordinary atomic A2A
    enqueue records its run. Thus a crash or an approval block between answer
    acceptance and enqueue cannot either lose the answer or repeat its effect.
    """
    async with service.store.connection() as db:
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            "SELECT * FROM api_a2a_answer_receipts WHERE owner=? AND message_id=?",
            (owner, message_id),
        ) as cursor:
            existing = await cursor.fetchone()
        if existing:
            if existing["digest"] != digest:
                raise InvalidParamsError("messageId was already used for different content")
            return dict(existing)
        async with db.execute(
            "SELECT * FROM api_a2a_messages WHERE owner=? AND message_id=?",
            (owner, message_id),
        ) as cursor:
            bound = await cursor.fetchone()
        if bound is not None:
            if bound["digest"] != digest:
                raise InvalidParamsError("messageId was already used for different content")
            return {**dict(bound), "ready": 0}
        async with db.execute(
            "SELECT t.id FROM api_a2a_tasks t JOIN api_runs r ON r.id=t.run_id WHERE t.id=? AND t.owner=? AND t.run_id=? AND r.state='paused'",
            (task_id, owner, run_id),
        ) as cursor:
            current = await cursor.fetchone()
        if not current:
            raise InvalidParamsError("The task has already continued; fetch its current state")
        scope = json.dumps(
            record.scope.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        async with db.execute(
            "SELECT c.payload FROM clarifications c JOIN sessions s ON s.id=c.session_id WHERE c.id=? AND c.session_id=? AND c.scope=? AND c.applied=0 AND json_extract(s.metadata,'$.pending_question_id')=c.id",
            (record.id, record.session_id, scope),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise InvalidParamsError("This clarification is no longer pending in the owned session")
        current_record = PendingQuestion.model_validate_json(row["payload"])
        if current_record.status == "pending" and current_record.expires_at <= datetime.now(UTC):
            current_record.status = "expired"
        if current_record.status == "pending":
            try:
                answers = parse_question_answer(current_record, text)
                current_record = apply_question_answers(current_record, answers)
            except (ValueError, TypeError) as exc:
                raise InvalidParamsError(f"Invalid clarification answer: {exc}") from None
        receipt = {
            "owner": owner,
            "message_id": message_id,
            "digest": digest,
            "task_id": task_id,
            "run_id": run_id,
            "question_id": current_record.id,
            "ready": int(current_record.status != "pending"),
        }
        await db.execute(
            "UPDATE clarifications SET payload=?,status=? WHERE id=?",
            (current_record.model_dump_json(), current_record.status, current_record.id),
        )
        await db.execute(
            "INSERT INTO api_a2a_answer_receipts VALUES (?,?,?,?,?,?,?)",
            tuple(receipt.values()),
        )
        await db.commit()
        return receipt
