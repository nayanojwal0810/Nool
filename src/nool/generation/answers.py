"""Answer company questions from current, authorized evidence."""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from nool.data.documents import DocumentStore
from nool.retrieval.company import (
    DEMO_USERS,
    CompanyHybridSearch,
    CompanySearchResult,
    corpus_fingerprint,
)

NO_EVIDENCE = (
    "I couldn't verify an answer from the documents available to you. "
    "Please check with the relevant team before acting on it."
)
PARTIAL_NOTE = "I couldn't verify the remaining details from the available documents."
UPDATING = (
    "The documents are being updated. Please try again after search is refreshed."
)
MODEL_UNAVAILABLE = (
    "I couldn't complete the document check right now. Please try again shortly."
)


@dataclass(frozen=True)
class Citation:
    document_id: str
    title: str
    chunk_id: str
    version: int
    section: str
    page_number: int | None
    quote: str

    @property
    def label(self) -> str:
        location = f"p. {self.page_number}" if self.page_number else self.section
        return f"{self.title} ({self.document_id}, {location}, v{self.version})"


@dataclass(frozen=True)
class AnswerResult:
    status: str
    text: str
    citations: tuple[Citation, ...] = ()


class AnswerModel(Protocol):
    def generate(self, question: str, evidence: list[CompanySearchResult]) -> dict: ...


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class CompanyAnswerService:
    """Build cited answers and refuse unsupported or stale evidence."""

    def __init__(
        self,
        search: CompanyHybridSearch,
        store: DocumentStore,
        model: AnswerModel,
    ) -> None:
        self.search = search
        self.store = store
        self.model = model

    def _evidence_is_current(
        self, evidence: list[CompanySearchResult], user_id: str
    ) -> bool:
        if corpus_fingerprint(self.store) != self.search.indexed_fingerprint:
            return False
        role = DEMO_USERS[user_id]
        for item in evidence:
            document = self.store.get_document(item.document_id)
            if (
                document is None
                or not document.active
                or role not in document.allowed_roles
                or document.version != item.version
                or document.title != item.title
            ):
                return False
            if not any(
                chunk.chunk_id == item.chunk_id and chunk.text == item.text
                for chunk in self.store.current_chunks(item.document_id)
            ):
                return False
        return True

    def answer(self, question: str, user_id: str, *, top_k: int = 5) -> AnswerResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")
        if user_id not in DEMO_USERS:
            raise ValueError("Unknown demo user")
        try:
            evidence = self.search.search(question.strip(), user_id, top_k=top_k)
        except RuntimeError as exc:
            if "outdated" in str(exc):
                return AnswerResult("unavailable", UPDATING)
            raise
        if not evidence:
            return AnswerResult("unsupported", NO_EVIDENCE)
        if corpus_fingerprint(self.store) != self.search.indexed_fingerprint:
            return AnswerResult("unavailable", UPDATING)
        if not self._evidence_is_current(evidence, user_id):
            return AnswerResult("unsupported", NO_EVIDENCE)

        try:
            response = self.model.generate(question.strip(), evidence)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            return AnswerResult("unavailable", MODEL_UNAVAILABLE)
        if corpus_fingerprint(self.store) != self.search.indexed_fingerprint:
            return AnswerResult("unavailable", UPDATING)
        if not self._evidence_is_current(evidence, user_id):
            return AnswerResult("unsupported", NO_EVIDENCE)
        if not isinstance(response, dict):
            return AnswerResult("unsupported", NO_EVIDENCE)
        status = response.get("status")
        if status == "unsupported":
            return AnswerResult("unsupported", NO_EVIDENCE)
        if status not in {"answered", "partial"}:
            return AnswerResult("unsupported", NO_EVIDENCE)
        answer = response.get("answer")
        references = response.get("citations")
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or len(answer) > 800
            or not isinstance(references, list)
            or not references
            or len(references) > len(evidence)
        ):
            return AnswerResult("unsupported", NO_EVIDENCE)

        by_id = {item.chunk_id: item for item in evidence}
        citations = []
        seen: set[str] = set()
        for reference in references:
            if not isinstance(reference, dict):
                return AnswerResult("unsupported", NO_EVIDENCE)
            chunk_id = reference.get("evidence_id")
            quote = reference.get("quote")
            if (
                not isinstance(chunk_id, str)
                or chunk_id not in by_id
                or chunk_id in seen
                or not isinstance(quote, str)
                or len(quote.strip()) < 8
                or _normalized(quote) not in _normalized(by_id[chunk_id].text)
            ):
                return AnswerResult("unsupported", NO_EVIDENCE)
            seen.add(chunk_id)
            item = by_id[chunk_id]
            citations.append(
                Citation(
                    item.document_id,
                    item.title,
                    item.chunk_id,
                    item.version,
                    item.section,
                    item.page_number,
                    quote.strip(),
                )
            )

        text = answer.strip()
        if status == "partial":
            text = f"{text} {PARTIAL_NOTE}"
        return AnswerResult(status, text, tuple(citations))
