"""Answer company questions from current, authorized evidence."""

import re
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

from nool.data.documents import DocumentStore
from nool.generation.evidence import select_answer_passages
from nool.generation.questions import explicit_question_parts
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


_APPROVAL_CONDITION = re.compile(
    r"\b(?:with|subject to)\s+([a-z][a-z -]{0,40}?\bapproval)\b",
    re.IGNORECASE,
)
_STORAGE_ACTION = re.compile(r"\b(?:stor\w*|sav\w*|record\w*)\b", re.IGNORECASE)
_STORAGE_LOCATION = re.compile(
    r"\b(?:in|on|at|within|through|to)\s+"
    r"(?:the|a|an)?\s*(?:[a-z-]+\s+){0,3}"
    r"(?:system|portal|folder|drive|case|file|repository|database|vault|cabinet)\b",
    re.IGNORECASE,
)


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

        requested_parts = explicit_question_parts(question)
        try:
            original = self.search.search(question.strip(), user_id, top_k=top_k)
            if requested_parts:
                part_hits = [
                    self.search.search(part, user_id, top_k=1)[0:1]
                    for part in requested_parts
                ]
                candidates = [*part_hits[0], *part_hits[1], *original]
                seen_documents: set[str] = set()
                evidence = []
                for item in candidates:
                    if item.document_id not in seen_documents:
                        seen_documents.add(item.document_id)
                        evidence.append(replace(item, rank=len(evidence) + 1))
                        if len(evidence) == top_k:
                            break
            else:
                evidence = original
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
        if requested_parts:
            parts = response.get("parts")
            if not isinstance(parts, list) or len(parts) != len(requested_parts):
                return finish("unsupported", NO_EVIDENCE, "invalid_response")
            part_answers = []
            part_references = []
            part_statuses = []
            for part in parts:
                if not isinstance(part, dict):
                    return finish("unsupported", NO_EVIDENCE, "invalid_response")
                part_status = part.get("status")
                part_answer = part.get("answer")
                references = part.get("citations")
                if part_status == "unsupported":
                    if part_answer or references:
                        return finish("unsupported", NO_EVIDENCE, "invalid_response")
                elif (
                    part_status not in {"answered", "partial"}
                    or not isinstance(part_answer, str)
                    or not part_answer.strip()
                    or not isinstance(references, list)
                    or not references
                ):
                    return finish("unsupported", NO_EVIDENCE, "invalid_response")
                else:
                    part_answers.append(part_answer.strip().rstrip(". "))
                    part_references.extend(references)
                part_statuses.append(part_status)
            if not part_answers:
                return finish("unsupported", NO_EVIDENCE, "model_unsupported")
            response = {
                "status": (
                    "answered"
                    if all(status == "answered" for status in part_statuses)
                    else "partial"
                ),
                "answer": ". ".join(part_answers) + ".",
                "citations": part_references,
            }
        elif "parts" in response:
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
            or len(references) > len(evidence) * (2 if requested_parts else 1)
        ):
            return finish("unsupported", NO_EVIDENCE, "invalid_response")

        by_id = {item.chunk_id: item for item in evidence}
        citations = []
        seen: set[str] = set()
        repaired_citation_id = False
        for reference in references:
            if not isinstance(reference, dict):
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
            chunk_id = reference.get("evidence_id")
            quote = reference.get("quote")
            if (
                not isinstance(chunk_id, str)
                or not isinstance(quote, str)
                or len(quote.strip()) < 8
            ):
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
            item = by_id.get(chunk_id)
            if item is None:
                matches = [
                    candidate
                    for candidate in evidence
                    if len(chunk_id) == len(candidate.chunk_id)
                    and sum(
                        left != right
                        for left, right in zip(
                            chunk_id, candidate.chunk_id, strict=True
                        )
                    )
                    == 1
                    and _normalized(quote) in _normalized(candidate.text)
                ]
                if len(matches) != 1:
                    return finish("unsupported", NO_EVIDENCE, "invalid_citation")
                item = matches[0]
                repaired_citation_id = True
            if _normalized(quote) not in _normalized(item.text):
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
            if item.chunk_id in seen:
                if requested_parts:
                    continue
                return finish("unsupported", NO_EVIDENCE, "invalid_citation")
            seen.add(item.chunk_id)
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

        if (
            question.casefold().lstrip().startswith("where")
            and _STORAGE_ACTION.search(question)
            and not any(_STORAGE_LOCATION.search(item.quote) for item in citations)
        ):
            return finish("unsupported", NO_EVIDENCE, "missing_location_evidence")
        text = answer.strip()
        for citation in citations:
            for match in _APPROVAL_CONDITION.finditer(citation.quote):
                condition = match.group(1).strip()
                if _normalized(condition) not in _normalized(text):
                    text = f"{text} {condition.capitalize()} is required."
        if status == "partial":
            text = f"{text} {PARTIAL_NOTE}"
        reason = "accepted_citation_id_repaired" if repaired_citation_id else "accepted"
        return finish(status, text, reason, tuple(citations))
