"""Owned clarification answers; answering never grants permission for an action."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from harness.core.clarification import PendingQuestion, QuestionStore
from harness.core.session_search import session_scope
from harness.server.models import ServiceError

if TYPE_CHECKING:
    from harness.core.schemas import Session
    from harness.server.service import HarnessService


class QuestionAnswers(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    answers: dict[str, str | list[str]] = Field(min_length=1, max_length=5)

    @field_validator("answers")
    @classmethod
    def bounded_answers(cls, answers):
        for key, value in answers.items():
            if key not in {"q0", "q1", "q2", "q3", "q4"}:
                raise ValueError("Answer keys must be q0 through q4")
            values = value if isinstance(value, list) else [value]
            if len(values) > 8 or any(len(item) > 8000 for item in values):
                raise ValueError("Answers exceed the question size limit")
        return answers


class QuestionManager:
    def __init__(self, service: HarnessService):
        self.service = service
        self.store: QuestionStore | None = None

    async def start(self):
        if self.store is None:
            self.store = await asyncio.to_thread(QuestionStore, self.service.database)

    async def close(self):
        if self.store is not None:
            await asyncio.to_thread(self.store.close)
            self.store = None

    async def check_admission(self, session: Session | None) -> None:
        if session is None or not session.metadata.get("pending_question_id"):
            return
        scope = session_scope(session)
        if scope is None or self.store is None:
            raise ServiceError(409, "The pending question cannot be resolved in this runtime")
        record = await asyncio.to_thread(
            self.store.get,
            str(session.metadata["pending_question_id"]),
            scope=scope,
            session_id=session.id,
        )
        if record is None:
            raise ServiceError(409, "Pending question state is missing; inspect this session")
        if record.status == "pending":
            raise ServiceError(409, "Answer or skip the pending questions before continuing")

    async def _owned(self, owner: str, question_id: str):
        rows = await self.service.store.rows(
            "SELECT c.session_id FROM clarifications c JOIN api_sessions s ON s.id=c.session_id WHERE c.id=? AND s.owner=?",
            (question_id, owner),
        )
        if not rows or self.store is None:
            raise ServiceError(404, "Question not found")
        session = await self.service._owned_session(owner, rows[0]["session_id"])
        scope = session_scope(session) if session is not None else None
        if session is None or scope is None:
            raise ServiceError(404, "Question not found")
        record = await asyncio.to_thread(
            self.store.get, question_id, scope=scope, session_id=session.id
        )
        if record is None:
            raise ServiceError(404, "Question not found")
        if session.metadata.get("pending_question_id") != record.id or record.applied:
            raise ServiceError(409, "This question is no longer the session's pending question")
        runs = await self.service.store.rows(
            "SELECT id,state FROM api_runs WHERE session_id=? AND owner=? ORDER BY created_at DESC,id DESC LIMIT 1",
            (session.id, owner),
        )
        return record, runs[0] if runs else None

    async def descriptor(self, owner: str, record: PendingQuestion, run=None) -> dict[str, Any]:
        from harness.server.service import public

        if run is None:
            rows = await self.service.store.rows(
                "SELECT id,state FROM api_runs WHERE session_id=? AND owner=? ORDER BY created_at DESC,id DESC LIMIT 1",
                (record.session_id, owner),
            )
            run = rows[0] if rows else None
        return public(
            {
                **record.model_dump(mode="json", exclude={"scope", "tool_call_id"}),
                "resume_run_id": run["id"] if run and run["state"] == "paused" else None,
            }
        )

    async def list(self, owner: str) -> list[dict[str, Any]]:
        rows = await self.service.store.rows(
            "SELECT c.id FROM clarifications c JOIN api_sessions s ON s.id=c.session_id WHERE s.owner=? AND c.applied=0 AND 'paused'=(SELECT r.state FROM api_runs r WHERE r.session_id=s.id ORDER BY r.created_at DESC,r.id DESC LIMIT 1) ORDER BY c.expires_at,c.id LIMIT 200",
            (owner,),
        )
        records = []
        for row in rows:
            try:
                record, run = await self._owned(owner, row["id"])
            except ServiceError as exc:
                if exc.status in {404, 409}:
                    continue
                raise
            if run and run["state"] == "paused":
                records.append(await self.descriptor(owner, record, run))
        return records

    async def for_session(self, owner: str, session_id: str) -> list[dict[str, Any]]:
        session = await self.service._owned_session(owner, session_id)
        question_id = session.metadata.get("pending_question_id") if session else None
        if not question_id:
            return []
        try:
            record, run = await self._owned(owner, str(question_id))
        except ServiceError as exc:
            if exc.status in {404, 409}:
                return []
            raise
        if not run or run["state"] != "paused":
            return []
        return [await self.descriptor(owner, record, run)]

    async def answer(self, owner: str, question_id: str, submission: QuestionAnswers):
        record, run = await self._owned(owner, question_id)
        if not run or run["state"] != "paused":
            raise ServiceError(409, "The question's run is not paused for input")
        assert self.store is not None
        try:
            answered = await asyncio.to_thread(
                self.store.answer,
                question_id,
                scope=record.scope,
                session_id=record.session_id,
                answers=submission.answers,
            )
        except ValueError as exc:
            raise ServiceError(409, str(exc)) from None
        return await self.descriptor(owner, answered, run)

    async def cancel(self, owner: str, question_id: str):
        record, run = await self._owned(owner, question_id)
        if not run or run["state"] != "paused":
            raise ServiceError(409, "The question's run is not paused for input")
        assert self.store is not None
        cancelled = await asyncio.to_thread(
            self.store.cancel, question_id, scope=record.scope, session_id=record.session_id
        )
        return await self.descriptor(owner, cancelled, run)
