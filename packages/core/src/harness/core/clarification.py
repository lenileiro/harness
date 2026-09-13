"""Durable human questions, separate from permission/approval decisions."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from harness.core.memory import MemoryScope

Answer = str | list[str]


class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    choices: list[str] | None = Field(default=None, max_length=4)
    multi_select: bool = False

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain text")
        return value.strip()

    @field_validator("choices")
    @classmethod
    def clean_choices(cls, values: list[str] | None) -> list[str] | None:
        if not values:
            return None
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 500 for value in cleaned) or len(set(cleaned)) != len(
            cleaned
        ):
            raise ValueError("choices must be distinct nonempty strings of at most 500 characters")
        return cleaned


class ClarifyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[QuestionSpec] = Field(min_length=1, max_length=5)


class PendingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: "question_" + uuid.uuid4().hex)
    session_id: str
    scope: MemoryScope
    tool_call_id: str
    questions: list[QuestionSpec] = Field(min_length=1, max_length=5)
    answers: dict[str, Answer] = Field(default_factory=dict)
    status: Literal["pending", "answered", "expired", "cancelled"] = "pending"
    expires_at: datetime
    applied: bool = False

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "responses": [
                {
                    "question": question.question,
                    "choices_offered": question.choices,
                    "user_response": self.answers.get(
                        f"q{index}", [] if question.multi_select and question.choices else ""
                    ),
                }
                for index, question in enumerate(self.questions)
            ]
        }
        if self.status == "expired":
            result["timed_out"] = True
        if self.status == "cancelled":
            result["cancelled"] = True
        return result


def render_question(record: PendingQuestion) -> str:
    lines = [f"Clarification required: {record.id}"]
    for index, question in enumerate(record.questions):
        key = f"q{index}"
        if key in record.answers:
            continue
        lines.append(f"{key}: {question.question}")
        if question.choices:
            for number, choice in enumerate(question.choices, 1):
                label = (
                    f"{choice} (Recommended)"
                    if number == 1 and len(question.choices) > 1
                    else choice
                )
                lines.append(f"  {number}. {label}")
            lines.append("  Other: type your answer.")
        if question.multi_select and question.choices:
            lines.append("  Multiple selections allowed; separate choices with commas.")
    lines.append(
        f"Reply `/answer {record.id} YOUR ANSWER` for the next question, or provide a JSON object keyed by q0, q1, ... for a batch."
    )
    lines.append(
        f"Expires: {record.expires_at.isoformat(timespec='seconds')}. This is a question, not an approval request."
    )
    return "\n".join(lines)


def parse_question_answer(record: PendingQuestion, text: str) -> dict[str, Answer]:
    """Text answers the next unanswered question; JSON supports a whole form."""
    if len(text) > 40000:
        raise ValueError("answer exceeds 40000 characters")
    if text.lstrip().startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("answers must be a JSON object")
        return parsed
    key = next(
        (f"q{i}" for i in range(len(record.questions)) if f"q{i}" not in record.answers), None
    )
    if key is None:
        raise ValueError("this question has already been answered")
    return {key: text}


def _answer_value(question: QuestionSpec, value: Any) -> Answer:
    multiple = bool(question.multi_select and question.choices)
    if isinstance(value, list):
        if not multiple or len(value) > 8 or any(not isinstance(v, str) for v in value):
            raise ValueError(
                "lists are valid only for multiple selection questions (maximum 8 answers)"
            )
        values = value
    elif isinstance(value, str):
        values = value.split(",") if multiple else [value]
    else:
        raise ValueError("answers must be strings or lists of strings")
    cleaned: list[str] = []
    for item in values:
        item = item.strip()
        if len(item) > 8000:
            raise ValueError("each answer must be at most 8000 characters")
        if question.choices and item.isascii() and item.isdigit():
            index = int(item) - 1
            if 0 <= index < len(question.choices):
                item = question.choices[index]
        if (not multiple or item) and item not in cleaned:
            cleaned.append(item)
    return cleaned if multiple else cleaned[0]


def apply_question_answers(record: PendingQuestion, answers: dict[str, Answer]) -> PendingQuestion:
    """Validate and merge answers without mutating the record or writing storage.

    Callers must establish the trusted owner/session before loading the record
    and persist this result in the same transaction as any delivery receipt.
    """
    if (
        record.status not in {"pending", "answered"}
        or record.applied
        or (record.status == "pending" and record.expires_at <= datetime.now(UTC))
    ):
        raise ValueError("question is expired, cancelled, or already consumed")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("provide at least one answer")
    updated = record.model_copy(deep=True)
    valid_keys = {f"q{i}" for i in range(len(updated.questions))}
    for key, value in answers.items():
        if key not in valid_keys:
            raise ValueError("unknown question key; expected q0, q1, ...")
        normalized = _answer_value(updated.questions[int(key[1:])], value)
        if key in updated.answers and updated.answers[key] != normalized:
            raise ValueError("an accepted answer cannot be changed")
        updated.answers[key] = normalized
    if len(updated.answers) == len(updated.questions):
        updated.status = "answered"
    return updated


class QuestionStore:
    """SQLite question ledger; use the session database path for shared ownership.

    Every write validates the caller's trusted scope and exact session. Instances
    own one connection and must be closed by their creating service/agent.
    """

    def __init__(self, path: str | Path, *, timeout_seconds: int = 900):
        if not 1 <= timeout_seconds <= 86400:
            raise ValueError("question timeout must be between 1 and 86400 seconds")
        self.path = str(path)
        self.timeout_seconds = timeout_seconds
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS clarifications (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, scope TEXT NOT NULL,
                tool_call_id TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                expires_at REAL NOT NULL, applied INTEGER NOT NULL DEFAULT 0,
                UNIQUE(session_id, tool_call_id, scope)
            );
            CREATE INDEX IF NOT EXISTS clarification_pending ON clarifications(scope, session_id, applied);
        """)

    def close(self) -> None:
        with self._lock:
            self.db.close()

    @staticmethod
    def _scope(scope: MemoryScope) -> str:
        return json.dumps(scope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def _save(self, record: PendingQuestion) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO clarifications VALUES (?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.session_id,
                self._scope(record.scope),
                record.tool_call_id,
                record.model_dump_json(),
                record.status,
                record.expires_at.timestamp(),
                int(record.applied),
            ),
        )

    def _load(self, row: sqlite3.Row) -> PendingQuestion:
        record = PendingQuestion.model_validate_json(row["payload"])
        if record.status == "pending" and record.expires_at <= datetime.now(UTC):
            record.status = "expired"
            self._save(record)
        return record

    def get(
        self, question_id: str, *, scope: MemoryScope, session_id: str | None = None
    ) -> PendingQuestion | None:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM clarifications WHERE id=? AND scope=?",
                (question_id, self._scope(scope)),
            ).fetchone()
            if row is None or (session_id is not None and row["session_id"] != session_id):
                return None
            return self._load(row)

    def list_pending(
        self, *, scope: MemoryScope, session_id: str | None = None
    ) -> list[PendingQuestion]:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            rows = self.db.execute(
                "SELECT * FROM clarifications WHERE scope=? AND applied=0 ORDER BY rowid",
                (self._scope(scope),),
            ).fetchall()
            return [
                self._load(row)
                for row in rows
                if session_id is None or row["session_id"] == session_id
            ]

    def create(
        self,
        *,
        session_id: str,
        scope: MemoryScope,
        tool_call_id: str,
        questions: list[QuestionSpec],
    ) -> PendingQuestion:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute(
                "SELECT * FROM clarifications WHERE session_id=? AND scope=? AND tool_call_id=?",
                (session_id, self._scope(scope), tool_call_id),
            ).fetchone()
            if existing:
                record = self._load(existing)
                if record.questions != questions:
                    raise ValueError("tool call already belongs to a different question")
                return record
            if self.db.execute(
                "SELECT 1 FROM clarifications WHERE session_id=? AND scope=? AND applied=0",
                (session_id, self._scope(scope)),
            ).fetchone():
                raise ValueError("answer the pending clarification before requesting another")
            record = PendingQuestion(
                session_id=session_id,
                scope=scope,
                tool_call_id=tool_call_id,
                questions=questions,
                expires_at=datetime.now(UTC) + timedelta(seconds=self.timeout_seconds),
            )
            self._save(record)
            return record

    def answer(
        self, question_id: str, *, scope: MemoryScope, session_id: str, answers: dict[str, Answer]
    ) -> PendingQuestion:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM clarifications WHERE id=? AND scope=? AND session_id=?",
                (question_id, self._scope(scope), session_id),
            ).fetchone()
            if row is None:
                raise ValueError("question not found for this session and owner")
            record = apply_question_answers(self._load(row), answers)
            self._save(record)
            return record

    def cancel(self, question_id: str, *, scope: MemoryScope, session_id: str) -> PendingQuestion:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM clarifications WHERE id=? AND scope=? AND session_id=?",
                (question_id, self._scope(scope), session_id),
            ).fetchone()
            if row is None:
                raise ValueError("question not found for this session and owner")
            record = self._load(row)
            if record.status in {"pending", "answered"} and not record.applied:
                record.status = "cancelled"
                self._save(record)
            return record

    def cancel_session(self, *, scope: MemoryScope, session_id: str) -> None:
        for record in self.list_pending(scope=scope, session_id=session_id):
            self.cancel(record.id, scope=scope, session_id=session_id)

    def mark_applied(self, question_id: str, *, scope: MemoryScope, session_id: str) -> None:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM clarifications WHERE id=? AND scope=? AND session_id=?",
                (question_id, self._scope(scope), session_id),
            ).fetchone()
            if row is None:
                raise ValueError("question not found for this session and owner")
            record = self._load(row)
            if record.status == "pending":
                raise ValueError("pending question cannot be consumed")
            record.applied = True
            self._save(record)
