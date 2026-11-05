"""Answer company questions from current, authorized evidence."""

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from nool.data.documents import DocumentStore
from nool.generation.evidence import select_answer_passages
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
PARTIAL_NOTE = "I can only confirm the details above from the available documents."
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
    reason_code: str = ""
    retrieved_document_ids: tuple[str, ...] = ()
    evidence_chunk_ids: tuple[str, ...] = ()
    model_duration_ms: int | None = None


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
        retrieved_document_ids: tuple[str, ...] = ()
        evidence_chunk_ids: tuple[str, ...] = ()
        model_duration_ms: int | None = None

        def finish(
            status: str,
            message: str,
            reason_code: str,
            citations: tuple[Citation, ...] = (),
        ) -> AnswerResult:
            return AnswerResult(
                status,
                message,
                citations,
                reason_code,
                retrieved_document_ids,
                evidence_chunk_ids,
                model_duration_ms,
            )

        try:
            evidence = self.search.search(question.strip(), user_id, top_k=top_k)
        except RuntimeError as exc:
            if "outdated" in str(exc):
                return finish("unavailable", UPDATING, "stale_index")
            raise
        retrieved_document_ids = tuple(item.document_id for item in evidence)
        if not evidence:
            return finish("unsupported", NO_EVIDENCE, "no_search_results")
        if corpus_fingerprint(self.store) != self.search.indexed_fingerprint:
            return finish("unavailable", UPDATING, "stale_index")
        if not self._evidence_is_current(evidence, user_id):
            return finish("unsupported", NO_EVIDENCE, "evidence_changed")
        evidence = select_answer_passages(question, evidence, self.store)
        evidence_chunk_ids = tuple(item.chunk_id for item in evidence)
        if not evidence:
            return finish("unsupported", NO_EVIDENCE, "no_valid_passages")
        if not self._evidence_is_current(evidence, user_id):
            return finish("unsupported", NO_EVIDENCE, "evidence_changed")

        started = perf_counter()
        try:
            response = self.model.generate(question.strip(), evidence)
        except TimeoutError:
            model_duration_ms = round((perf_counter() - started) * 1000)
            return finish("unavailable", MODEL_UNAVAILABLE, "model_timeout")
        except OSError:
            model_duration_ms = round((perf_counter() - started) * 1000)
            return finish("unavailable", MODEL_UNAVAILABLE, "model_transport_error")
        except ValueError:
            model_duration_ms = round((perf_counter() - started) * 1000)
            return finish("unavailable", MODEL_UNAVAILABLE, "model_invalid_output")
        model_duration_ms = round((perf_counter() - started) * 1000)
        if corpus_fingerprint(self.store) != self.search.indexed_fingerprint:
            return finish("unavailable", UPDATING, "stale_index")
        if not self._evidence_is_current(evidence, user_id):
            return finish("unsupported", NO_EVIDENCE, "evidence_changed")
        if not isinstance(response, dict):
            return finish("unsupported", NO_EVIDENCE, "invalid_response")
        status = response.get("status")
        if status == "unsupported":
            return finish("unsupported", NO_EVIDENCE, "model_unsupported")
        if status not in {"answered", "partial"}:
            return finish("unsupported", NO_EVIDENCE, "invalid_response")
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
            return finish("unsupported", NO_EVIDENCE, "invalid_response")

        by_id = {item.chunk_id: item for item in evidence}
        citations = []
        seen: set[str] = set()
        for reference in references:
            if not isinstance(reference, dict):
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
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
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
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
        return finish(status, text, "accepted", tuple(citations))
