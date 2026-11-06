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
from nool.generation.evidence import select_answer_passages
from nool.generation.ollama import OllamaAnswerModel
from nool.generation.questions import explicit_question_parts, focused_question
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What details go into the decision record, and when is it saved?",
            ("What details go into the decision record", "when is it saved"),
        ),
        (
            "How should I handle the visitor and report the repair?",
            ("handle the visitor", "report the repair"),
        ),
        ("Who needs manager and Finance approval?", ()),
        ("How should I plan leave around holidays?", ()),
    ],
)
def test_explicit_question_parts(question: str, expected: tuple[str, ...]) -> None:
    assert explicit_question_parts(question) == expected


def test_focused_question_keeps_the_request_after_context() -> None:
    assert (
        focused_question(
            "After an approved dinner, what is the expense claim deadline?"
        )
        == "what is the expense claim deadline?"
    )
    assert focused_question("How much leave do I get?") == "How much leave do I get?"


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
    assert result.reason_code == "accepted"
    assert result.retrieved_document_ids == ("PEO-001",)
    assert hit.chunk_id in result.evidence_chunk_ids


def test_unique_quote_repairs_one_character_id_and_keeps_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flexible.md"
    source.write_text(
        "# Flexible Work Policy\n\n## Policy\n\n"
        "Eligible employees may work remotely up to two days per week "
        "with manager approval.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="PEO-004",
        title="Flexible Work Policy",
        owner="People",
        allowed_roles=["employee"],
    )
    chunk = store.current_chunks("PEO-004")[1]
    hit = CompanySearchResult(
        "PEO-004",
        "Flexible Work Policy",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        1.0,
        1,
        ("bm25",),
    )
    first = "0" if chunk.chunk_id[0] != "0" else "1"
    response = {
        "status": "answered",
        "answer": "Eligible employees may work remotely up to two days per week.",
        "citations": [
            {
                "evidence_id": first + chunk.chunk_id[1:],
                "quote": chunk.text,
            }
        ],
    }
    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(response)
    ).answer("How many remote days are available?", "employee_demo")
    assert result.status == "answered"
    assert result.reason_code == "accepted_citation_id_repaired"
    assert result.citations[0].chunk_id == chunk.chunk_id
    assert "Manager approval is required." in result.text


def test_policy_passage_is_available_when_search_finds_another_section(
    tmp_path: Path,
) -> None:
    source = tmp_path / "holiday.md"
    source.write_text(
        "# Holiday Calendar\n\n## Policy\n\nThe calendar is published by 1 December."
        "\n\n## Procedure\n\nCheck the published calendar before travel.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="PEO-003",
        title="Holiday Calendar",
        owner="People",
        allowed_roles=["employee"],
    )
    chunks = {chunk.section: chunk for chunk in store.current_chunks("PEO-003")}
    procedure = chunks["Procedure"]
    policy = chunks["Policy"]
    hit = CompanySearchResult(
        "PEO-003",
        "Holiday Calendar",
        procedure.chunk_id,
        procedure.section,
        procedure.page_number,
        procedure.version,
        procedure.text,
        1.0,
        1,
        ("bm25",),
    )

    class PolicyModel:
        def generate(self, question: str, evidence: list) -> dict:
            assert policy.chunk_id in {item.chunk_id for item in evidence}
            return {
                "status": "answered",
                "answer": "The calendar is published by 1 December.",
                "citations": [
                    {
                        "evidence_id": policy.chunk_id,
                        "quote": "The calendar is published by 1 December.",
                    }
                ],
            }

    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, PolicyModel()
    ).answer("When is the holiday calendar published?", "employee_demo")
    assert result.status == "answered"
    assert result.citations[0].section == "Policy"


def test_exception_passage_is_available_beside_the_policy(tmp_path: Path) -> None:
    source = tmp_path / "travel.md"
    source.write_text(
        "# Travel Booking\n\n## Policy\n\n"
        "Request travel seven calendar days before departure.\n\n"
        "## Exceptions\n\nUrgent travel can use shorter notice with "
        "written manager and Finance approval.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="FIN-002",
        title="Travel Booking",
        owner="Finance",
        allowed_roles=["employee"],
    )
    chunks = {chunk.section: chunk for chunk in store.current_chunks("FIN-002")}
    policy = chunks["Policy"]
    exception = chunks["Exceptions"]
    hit = CompanySearchResult(
        "FIN-002",
        "Travel Booking",
        policy.chunk_id,
        policy.section,
        policy.page_number,
        policy.version,
        policy.text,
        1.0,
        1,
        ("bm25",),
    )

    class ExceptionModel:
        def generate(self, question: str, evidence: list) -> dict:
            assert exception.chunk_id in {item.chunk_id for item in evidence}
            return {
                "status": "answered",
                "answer": "Urgent travel needs written manager and Finance approval.",
                "citations": [
                    {
                        "evidence_id": exception.chunk_id,
                        "quote": exception.text,
                    }
                ],
            }

    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ExceptionModel()
    ).answer("What approval allows shorter travel notice?", "employee_demo")
    assert result.status == "answered"
    assert result.citations[0].section == "Exceptions"


def test_retrieved_procedure_is_kept_for_a_second_ranked_document(
    tmp_path: Path,
) -> None:
    source = tmp_path / "software.md"
    source.write_text(
        "# Software Guide\n\n## Policy\n\nOnly approved software may be used.\n\n"
        "## Procedure\n\nSend the missing tool name and vendor to the service desk.\n\n"
        "## Exceptions\n\nA missing tool may require additional review.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="IT-007",
        title="Software Guide",
        owner="IT",
        allowed_roles=["employee"],
    )
    procedure = next(
        chunk
        for chunk in store.current_chunks("IT-007")
        if chunk.section == "Procedure"
    )
    hit = CompanySearchResult(
        "IT-007",
        "Software Guide",
        procedure.chunk_id,
        procedure.section,
        procedure.page_number,
        procedure.version,
        procedure.text,
        0.8,
        2,
        ("contriever",),
    )
    selected = select_answer_passages(
        "How do I request review for a missing software tool?", [hit], store
    )
    assert procedure.chunk_id in {item.chunk_id for item in selected}


def test_example_does_not_displace_exception_or_policy(tmp_path: Path) -> None:
    source = tmp_path / "visitors.md"
    source.write_text(
        "# Visitor Access\n\n## Policy\n\nRegister visitors before arrival.\n\n"
        "## Example\n\nA customer visits reception tomorrow.\n\n"
        "## Exceptions\n\nFor an unplanned visitor, reception waits for host approval.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="OPS-004",
        title="Visitor Access",
        owner="Operations",
        allowed_roles=["employee"],
    )
    chunks = {chunk.section: chunk for chunk in store.current_chunks("OPS-004")}
    policy = chunks["Policy"]
    hit = CompanySearchResult(
        "OPS-004",
        "Visitor Access",
        policy.chunk_id,
        policy.section,
        policy.page_number,
        policy.version,
        policy.text,
        1.0,
        1,
        ("bm25",),
    )

    class ExceptionModel:
        def generate(self, question: str, evidence: list) -> dict:
            assert chunks["Exceptions"].chunk_id in {item.chunk_id for item in evidence}
            assert chunks["Example"].chunk_id not in {
                item.chunk_id for item in evidence
            }
            return {
                "status": "answered",
                "answer": "Reception waits for host approval.",
                "citations": [
                    {
                        "evidence_id": chunks["Exceptions"].chunk_id,
                        "quote": "reception waits for host approval",
                    }
                ],
            }

    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ExceptionModel()
    ).answer("Can an unplanned visitor enter reception?", "employee_demo")
    assert result.status == "answered"


def test_reporting_question_selects_the_action_passage(tmp_path: Path) -> None:
    source = tmp_path / "conduct.md"
    source.write_text(
        "# Workplace Conduct Policy\n\n"
        "## Procedure\n\nRaise a concern with a manager or People Operations. "
        "Contact People Operations through the confidential channel.\n\n"
        "## Responsibilities and follow-up\n\nPeople Operations records "
        "reported concerns in a restricted case file.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="PEO-005",
        title="Workplace Conduct Policy",
        owner="People",
        allowed_roles=["employee"],
    )
    chunks = {chunk.section: chunk for chunk in store.current_chunks("PEO-005")}
    follow_up = chunks["Responsibilities and follow-up"]
    procedure = chunks["Procedure"]
    hit = CompanySearchResult(
        "PEO-005",
        "Workplace Conduct Policy",
        follow_up.chunk_id,
        follow_up.section,
        follow_up.page_number,
        follow_up.version,
        follow_up.text,
        1.0,
        1,
        ("bm25",),
    )

    class ActionModel:
        def generate(self, question: str, evidence: list) -> dict:
            assert evidence[0].chunk_id == procedure.chunk_id
            return {
                "status": "answered",
                "answer": "Report to a manager or People Operations.",
                "citations": [
                    {
                        "evidence_id": procedure.chunk_id,
                        "quote": "Raise a concern with a manager or People Operations.",
                    }
                ],
            }

    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ActionModel()
    ).answer(
        "Where can an employee report a workplace conduct concern?", "employee_demo"
    )
    assert result.status == "answered"
    assert result.citations[0].section == "Procedure"


def test_storage_question_needs_a_cited_destination(tmp_path: Path) -> None:
    source = tmp_path / "incident.md"
    source.write_text(
        "# Incident Guide\n\n## Policy\n\n"
        "Preserve messages until the response team advises otherwise.",
        encoding="utf-8",
    )
    store = DocumentStore(tmp_path / "documents.sqlite")
    store.ingest(
        source,
        document_id="IT-005",
        title="Incident Guide",
        owner="IT",
        allowed_roles=["employee"],
    )
    chunk = store.current_chunks("IT-005")[1]
    hit = CompanySearchResult(
        "IT-005",
        "Incident Guide",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        1.0,
        1,
        ("bm25",),
    )
    response = {
        "status": "answered",
        "answer": "Preserve messages until advised otherwise.",
        "citations": [
            {
                "evidence_id": chunk.chunk_id,
                "quote": "Preserve messages until the response team advises otherwise.",
            }
        ],
    }
    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(response)
    ).answer("Where should investigation evidence be stored?", "employee_demo")
    assert result.status == "unsupported"
    assert result.reason_code == "missing_location_evidence"


def test_one_source_can_answer_two_parts_when_it_contains_both_facts(
    tmp_path: Path,
) -> None:
    _, store, leave = setup_policy(tmp_path)
    source = tmp_path / "holiday.md"
    source.write_text(
        "# Holiday Calendar\n\n## Policy\n\nUse the calendar for your location.",
        encoding="utf-8",
    )
    store.ingest(
        source,
        document_id="PEO-003",
        title="Holiday Calendar",
        owner="People",
        allowed_roles=["employee"],
    )
    chunk = store.current_chunks("PEO-003")[1]
    holiday = CompanySearchResult(
        "PEO-003",
        "Holiday Calendar",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        0.9,
        2,
        ("contriever",),
    )
    response = {"parts": [supported(leave), supported(leave)]}
    result = CompanyAnswerService(
        SearchStub(store, [leave, holiday]), store, ModelStub(response)
    ).answer("Which employees receive leave and how many days?", "employee_demo")
    assert result.status == "answered"
    assert result.citations[0].document_id == "PEO-001"
    bad_second = supported(leave)
    bad_second["citations"][0]["quote"] = "Invented leave rule."
    rejected = CompanyAnswerService(
        SearchStub(store, [leave, holiday]),
        store,
        ModelStub({"parts": [supported(leave), bad_second]}),
    ).answer("Which employees receive leave and how many days?", "employee_demo")
    assert rejected.reason_code == "invalid_citation"


def test_two_parts_require_valid_evidence_from_each_document(tmp_path: Path) -> None:
    _, store, leave = setup_policy(tmp_path)
    source = tmp_path / "holiday.md"
    source.write_text(
        "# Holiday Calendar\n\n## Policy\n\n"
        "The holiday calendar is published by 1 December.",
        encoding="utf-8",
    )
    store.ingest(
        source,
        document_id="PEO-003",
        title="Holiday Calendar",
        owner="People",
        allowed_roles=["employee"],
    )
    chunk = store.current_chunks("PEO-003")[1]
    holiday = CompanySearchResult(
        "PEO-003",
        "Holiday Calendar",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        0.9,
        2,
        ("contriever",),
    )
    holiday_part = {
        "status": "answered",
        "answer": "The holiday calendar is published by 1 December.",
        "citations": [
            {
                "evidence_id": holiday.chunk_id,
                "quote": "The holiday calendar is published by 1 December.",
            }
        ],
    }
    question = "How much paid leave do I get and when is the calendar published?"

    class PartSearch(SearchStub):
        def search(self, question: str, user_id: str, top_k: int = 5) -> list:
            if question.startswith("when"):
                return [holiday]
            return [leave]

    search = PartSearch(store, [leave])
    complete = CompanyAnswerService(
        search, store, ModelStub({"parts": [supported(leave), holiday_part]})
    ).answer(question, "employee_demo")
    assert complete.status == "answered"
    assert {item.document_id for item in complete.citations} == {"PEO-001", "PEO-003"}
    assert complete.retrieved_document_ids == ("PEO-001", "PEO-003")

    invalid_part = {
        **holiday_part,
        "citations": [{"evidence_id": holiday.chunk_id, "quote": "Invented date."}],
    }
    invalid = CompanyAnswerService(
        search, store, ModelStub({"parts": [supported(leave), invalid_part]})
    ).answer(question, "employee_demo")
    assert invalid.status == "unsupported"
    assert invalid.reason_code == "invalid_citation"
    assert invalid.citations == ()


def test_focused_search_recovers_a_document_missed_by_context_search(
    tmp_path: Path,
) -> None:
    _, store, leave = setup_policy(tmp_path)
    source = tmp_path / "meeting.md"
    source.write_text(
        "# Meeting Guide\n\n## Policy\n\nRecord meeting decisions.",
        encoding="utf-8",
    )
    store.ingest(
        source,
        document_id="OPS-001",
        title="Meeting Guide",
        owner="Operations",
        allowed_roles=["employee"],
    )
    chunk = store.current_chunks("OPS-001")[1]
    meeting = CompanySearchResult(
        "OPS-001",
        "Meeting Guide",
        chunk.chunk_id,
        chunk.section,
        chunk.page_number,
        chunk.version,
        chunk.text,
        1.0,
        1,
        ("bm25",),
    )

    class FocusSearch(SearchStub):
        def search(self, question: str, user_id: str, top_k: int = 5) -> list:
            return [leave] if question.startswith("how much") else [meeting]

    result = CompanyAnswerService(
        FocusSearch(store, [meeting]), store, ModelStub(supported(leave))
    ).answer("After a team meeting, how much paid leave do I get?", "employee_demo")
    assert result.status == "answered"
    assert result.retrieved_document_ids == ("OPS-001", "PEO-001")


def test_partial_and_unsupported_answers_use_professional_language(
    tmp_path: Path,
) -> None:
    _, store, hit = setup_policy(tmp_path)
    response = {
        "parts": [
            supported(hit),
            {"status": "unsupported", "answer": "", "citations": []},
        ]
    }
    partial = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(response)
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
    assert empty.reason_code == "no_search_results"
    assert stale.reason_code == "stale_index"
    assert model.calls == 0


def test_model_failure_uses_neutral_message(tmp_path: Path) -> None:
    _, store, hit = setup_policy(tmp_path)
    result = CompanyAnswerService(
        SearchStub(store, [hit]), store, ModelStub(OSError("model offline"))
    ).answer("How much leave?", "employee_demo")
    assert result.text == MODEL_UNAVAILABLE
    assert result.citations == ()
    assert result.reason_code == "model_transport_error"


def test_ollama_request_contains_only_given_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, hit = setup_policy(tmp_path)
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured["payload"] = json.loads(request.data)
            part = supported(hit)
            part["citations"][0]["evidence_id"] = "E1"
            request_body = json.loads(captured["payload"]["messages"][1]["content"])
            response = (
                {"parts": [part, part]} if request_body["requested_parts"] else part
            )
            return io.BytesIO(
                json.dumps({"message": {"content": json.dumps(response)}}).encode()
            )

    monkeypatch.setattr("nool.generation.ollama.build_opener", lambda proxy: Opener())
    response = OllamaAnswerModel().generate("How much leave?", [hit])
    assert response == supported(hit)
    request = json.loads(captured["payload"]["messages"][1]["content"])
    assert [item["evidence_id"] for item in request["evidence"]] == ["E1"]
    assert request["requested_parts"] == []
    assert "citations" in captured["payload"]["format"]["required"]
    response = OllamaAnswerModel().generate(
        "A visitor notices a broken door. How should I handle the visitor "
        "and report the repair?",
        [hit],
    )
    request = json.loads(captured["payload"]["messages"][1]["content"])
    assert request["requested_parts"] == [
        "handle the visitor",
        "report the repair",
    ]
    assert [part["citations"][0]["evidence_id"] for part in response["parts"]] == [
        hit.chunk_id,
        hit.chunk_id,
    ]
    assert captured["payload"]["format"]["properties"]["parts"]["minItems"] == 2
    OllamaAnswerModel(cpu_only=True).generate("How much leave?", [hit])
    assert captured["payload"]["options"]["num_gpu"] == 0
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
