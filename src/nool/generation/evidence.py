"""Choose concise passages from authorized, ranked company documents."""

import math
import re
from collections import Counter

import Stemmer

from nool.data.documents import DocumentStore
from nool.retrieval.company import CompanySearchResult

_STEMMER = Stemmer.Stemmer("english")
_WORDS = re.compile(r"[a-z][a-z0-9]+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_SECTION_LINE = re.compile(
    r"(?im)^[ \t]*(?:purpose and scope|policy|procedure|"
    r"responsibilities and follow-up|example|exceptions)[ \t]*$"
)
_SECTION_PREFIX = re.compile(
    r"^(?:policy|procedure|responsibilities and follow-up|exceptions|example)\s+",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _terms(text: str) -> set[str]:
    words = [
        word for word in _WORDS.findall(text.casefold()) if word not in _STOP_WORDS
    ]
    return set(_STEMMER.stemWords(words))


def select_citation_quote(passage: str, question: str, answer: str) -> str:
    """Copy a relevant sentence from a cited passage."""
    passage = _SECTION_LINE.sub("\n\n", passage)
    sentences = [
        " ".join(sentence.split())
        for sentence in _SENTENCE_BOUNDARY.split(passage)
        if len(sentence.strip()) >= 8
    ]
    if not sentences:
        return " ".join(passage.split())
    answer_terms = _terms(answer)
    question_terms = _terms(question)
    ranked = []
    sentence_terms = []
    for index, sentence in enumerate(sentences):
        terms = _terms(sentence)
        sentence_terms.append(terms)
        score = 2 * len(terms & answer_terms) + len(terms & question_terms)
        ranked.append((score, -index, index))
    best = max(ranked)[2]
    neighbors = [index for index in (best - 1, best + 1) if 0 <= index < len(sentences)]
    if neighbors:
        added = max(
            neighbors,
            key=lambda index: len(
                (sentence_terms[index] & answer_terms) - sentence_terms[best]
            ),
        )
        new_terms = (sentence_terms[added] & answer_terms) - sentence_terms[best]
        if new_terms and len((sentences[best] + sentences[added]).split()) <= 60:
            first, last = sorted((best, added))
            return _SECTION_PREFIX.sub("", " ".join(sentences[first : last + 1]))
    return _SECTION_PREFIX.sub("", sentences[best])


def select_answer_passages(
    question: str,
    ranked_documents: list[CompanySearchResult],
    store: DocumentStore,
) -> list[CompanySearchResult]:
    """Keep the ranked documents and choose their most relevant current passages."""
    candidates = [
        (document, chunk)
        for document in ranked_documents
        for chunk in store.current_chunks(document.document_id)
    ]
    question_terms = _terms(question)
    passage_terms = {chunk.chunk_id: _terms(chunk.text) for _, chunk in candidates}
    frequency = Counter(
        term for terms in passage_terms.values() for term in terms & question_terms
    )
    count = len(candidates)
    question_lower = question.casefold().strip()
    asks_for_number = question_lower.startswith(("how many", "how much", "how long"))
    asks_for_time = question_lower.startswith(("when", "by when"))
    asks_for_action = question_lower.startswith(("where", "how do", "how can"))

    def score(chunk) -> float:
        overlap = question_terms & passage_terms[chunk.chunk_id]
        value = sum(
            math.log((count + 1) / (frequency[term] + 1)) + 1 for term in overlap
        )
        if (asks_for_number or asks_for_time) and _NUMBER.search(chunk.text):
            value += 1
        if asks_for_action and chunk.section == "Procedure" and overlap:
            value += 2.5
        if chunk.section == "Introduction":
            value *= 0.2
        elif chunk.section in {"Purpose and scope", "Example"}:
            value *= 0.65
        elif chunk.section == "Responsibilities and follow-up":
            value *= 0.85
        return value

    selected = []
    for document in ranked_documents:
        chunks = store.current_chunks(document.document_id)
        ordered = sorted(
            chunks,
            key=lambda chunk: (score(chunk), chunk.chunk_id == document.chunk_id),
            reverse=True,
        )
        if not ordered:
            continue
        fallback = next(
            (chunk for chunk in chunks if chunk.chunk_id == document.chunk_id), None
        )
        best = ordered[0] if score(ordered[0]) > 0 else fallback
        if best is None:
            continue
        chosen = [best]
        if document.rank <= 2:
            substantive = (
                chunk
                for chunk in [fallback, *ordered]
                if chunk is not None
                and chunk.chunk_id != best.chunk_id
                and chunk.section
                not in {"Introduction", "Purpose and scope", "Example"}
                and score(chunk) > 0
            )
            second = next(substantive, None)
            if second is not None:
                chosen.append(second)
        if document.rank == 1:
            question_numbers = set(_NUMBER.findall(question))
            third = next(
                (
                    chunk
                    for chunk in ordered
                    if chunk.chunk_id not in {item.chunk_id for item in chosen}
                    and chunk.section not in {"Introduction", "Purpose and scope"}
                    and (
                        chunk.section != "Example"
                        or bool(question_numbers & set(_NUMBER.findall(chunk.text)))
                    )
                    and score(chunk) > 0
                ),
                None,
            )
            if third is not None:
                chosen.append(third)
        for chunk in chosen:
            selected.append(
                CompanySearchResult(
                    document.document_id,
                    document.title,
                    chunk.chunk_id,
                    chunk.section,
                    chunk.page_number,
                    chunk.version,
                    chunk.text,
                    document.score,
                    document.rank,
                    ("passage_selection",),
                )
            )
    return selected
