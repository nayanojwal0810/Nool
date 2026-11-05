"""Grounded answers, citations, and refusal behavior."""

import io
import json
from pathlib import Path

import pytest

from nool.data.documents import DocumentStore
from nool.generation.answers import (
    MODEL_UNAVAILABLE,
    NO_EVIDENCE,
    PARTIAL_NOTE,
    UPDATING,
    CompanyAnswerService,
)
from nool.generation.ollama import OllamaAnswerModel
from nool.retrieval.company import CompanySearchResult, corpus_fingerprint


class SearchStub:
    def __init__(
        self,
        store: DocumentStore,
        results: list[CompanySearchResult],
        *,
        stale: bool = False,
    ) -> None:
        self.results = results
        self.indexed_fingerprint = corpus_fingerprint(store)
        self.stale = stale

    def search(self, question: str, user_id: str, top_k: int = 5) -> list:
        if self.stale:
            raise RuntimeError("Company search index is outdated")
        return self.results[:top_k]


class ModelStub:
    def __init__(self, response: dict | Exception, callback=None) -> None:
        self.response = response
        self.callback = callback
        self.calls = 0

    def generate(self, question: str, evidence: list) -> dict:
        self.calls += 1
        if self.callback:
            self.callback()
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def setup_policy(tmp_path: Path, role: str = "employee") -> tuple:
    source = tmp_path / "leave.md"
    source.write_text(
        "# Leave Policy\n\n## Rule\n\nEmployees receive 20 paid leave days.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="PEO-001",
        title="Leave Policy",
        owner="People",
        allowed_roles=[role],
    )
    chunk = next(
        item
        for item in store.current_chunks("PEO-001")
        if "20 paid leave days" in item.text
    )
    hit = CompanySearchResult(
        "PEO-001",
        "Leave Policy",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        1.0,
        1,
        ("bm25", "contriever"),
    )
    return source, store, hit


def supported(hit: CompanySearchResult, status: str = "answered") -> dict:
    return {
        "status": status,
        "answer": "Employees receive 20 paid leave days.",
        "citations": [
            {
                "evidence_id": hit.chunk_id,
                "quote": "Employees receive 20 paid leave days.",
            }
        ],
    }


def test_supported_answer_has_traceable_citation(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path)
    model = ModelStub(supported(hit))
    service = CompanyAnswerService(SearchStub(store, [hit]), store, model)
    result = service.answer("How much leave?", "employee_demo")
    assert result.status == "answered"
    assert result.text == "Employees receive 20 paid leave days."
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "PEO-001"
    assert result.citations[0].quote in hit.text
    assert "Leave Policy" in result.citations[0].label


def test_partial_and_unsupported_answers_use_professional_language(
    tmp_path: Path,
) -> None:
    _, store, hit = setup_policy(tmp_path)
    partial = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(supported(hit, "partial"))
    ).answer("How much leave and what tax?", "employee_demo")
    assert partial.status == "partial"
    assert PARTIAL_NOTE in partial.text
    assert partial.citations

    unsupported = CompanyAnswerService(
        SearchStub(store, [hit]),
        store,
        ModelStub({"status": "unsupported", "answer": "A guess", "citations": []}),
    ).answer("What is next year's profit?", "employee_demo")
    assert unsupported.status == "unsupported"
    assert unsupported.text == NO_EVIDENCE
    assert unsupported.citations == ()


@pytest.mark.parametrize(
    "response",
    [
        {"status": "answered", "answer": "20 days", "citations": []},
        {
            "status": "answered",
            "answer": "20 days",
            "citations": [{"evidence_id": "unknown", "quote": "20 paid leave days"}],
        },
        {
            "status": "answered",
            "answer": "20 days",
            "citations": [{"evidence_id": "PLACEHOLDER", "quote": "A made-up rule"}],
        },
    ],
)
def test_invalid_citation_is_not_displayed(tmp_path: Path, response: dict) -> None:
    _, store, hit = setup_policy(tmp_path)
    if (
        response["citations"]
        and response["citations"][0]["evidence_id"] == "PLACEHOLDER"
    ):
        response["citations"][0]["evidence_id"] = hit.chunk_id
    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(response)
    ).answer("How much leave?", "employee_demo")
    assert result.text == NO_EVIDENCE
    assert result.citations == ()


def test_restricted_evidence_never_reaches_model(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path, "hr_admin")
    model = ModelStub(supported(hit))
    result = CompanyAnswerService(SearchStub(store, [hit]), store, model).answer(
        "How much leave?", "employee_demo"
    )
    assert result.text == NO_EVIDENCE
    assert model.calls == 0


def test_access_change_during_generation_removes_answer(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path)
    model = ModelStub(
        supported(hit),
        callback=lambda: store.set_allowed_roles("PEO-001", ["hr_admin"]),
    )
    result = CompanyAnswerService(SearchStub(store, [hit]), store, model).answer(
        "How much leave?", "employee_demo"
    )
    assert result.text == NO_EVIDENCE
    assert result.citations == ()


def test_content_change_during_generation_blocks_old_answer(tmp_path: Path) -> None:
    source, store, hit = setup_policy(tmp_path)

    def update() -> None:
        source.write_text("# Leave Policy\n\n## Rule\n\nEmployees receive 22 days.")
        store.ingest(
            source,
            document_id="PEO-001",
            title="Leave Policy",
            owner="People",
            allowed_roles=["employee"],
        )

    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(supported(hit), callback=update)
    ).answer("How much leave?", "employee_demo")
    assert result.status == "unavailable"
    assert result.text == UPDATING
    assert result.citations == ()


def test_empty_evidence_and_stale_index_skip_model(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path)
    model = ModelStub(supported(hit))
    empty = CompanyAnswerService(SearchStub(store, []), store, model).answer(
        "Unknown?", "employee_demo"
    )
    stale = CompanyAnswerService(
        SearchStub(store, [hit], stale=True), store, model
    ).answer("How much leave?", "employee_demo")
    assert empty.text == NO_EVIDENCE
    assert stale.text == UPDATING
    assert model.calls == 0


def test_model_failure_uses_neutral_message(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path)
    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(OSError("model offline"))
    ).answer("How much leave?", "employee_demo")
    assert result.text == MODEL_UNAVAILABLE
    assert result.citations == ()


def test_ollama_request_contains_only_given_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, hit = setup_policy(tmp_path)
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured["payload"] = json.loads(request.data)
            return io.BytesIO(
                json.dumps(
                    {"message": {"content": json.dumps(supported(hit))}}
                ).encode()
            )

    monkeypatch.setattr("nool.generation.ollama.build_opener", lambda proxy: Opener())
    response = OllamaAnswerModel().generate("How much leave?", [hit])
    assert response == supported(hit)
    request = json.loads(captured["payload"]["messages"][1]["content"])
    assert [item["evidence_id"] for item in request["evidence"]] == [hit.chunk_id]
    assert "citations" in captured["payload"]["format"]["required"]
    with pytest.raises(ValueError, match="local"):
        OllamaAnswerModel(base_url="https://remote.example")


def test_ollama_rejects_malformed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, hit = setup_policy(tmp_path)

    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(b"{}")

    monkeypatch.setattr("nool.generation.ollama.build_opener", lambda proxy: Opener())
    with pytest.raises(ValueError, match="invalid response"):
        OllamaAnswerModel().generate("How much leave?", [hit])
