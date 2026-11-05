"""Document ingestion and demo collection checks."""

import json
from collections import Counter
from pathlib import Path

import pytest
from pypdf import PdfWriter

from nool.config import get_project_root
from nool.data.collection import (
    ingest_collection,
    load_collection,
    validate_questions,
)
from nool.data.documents import (
    DocumentStore,
    Section,
    make_chunks,
    read_sections,
    source_digest,
)


def test_text_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_bytes(b"# Policy\r\n\r\nRule\r\n")
    first = source_digest(source)
    source.write_bytes(b"# Policy\n\nRule\n")
    assert source_digest(source) == first


def test_chunk_ids_are_stable_and_sections_remain_separate() -> None:
    sections = [Section("Rules", "one two three four five"), Section("Steps", "six")]
    first = make_chunks("DOC-1", 1, "hash", sections, words_per_chunk=3, overlap=1)
    second = make_chunks("DOC-1", 1, "hash", sections, words_per_chunk=3, overlap=1)
    assert first == second
    assert [item.section for item in first] == ["Rules", "Rules", "Steps"]
    assert len({item.chunk_id for item in first}) == 3
    assert make_chunks("DOC-1", 2, "hash", sections)[0].chunk_id != first[0].chunk_id


def test_document_update_access_and_archive(tmp_path: Path) -> None:
    source = tmp_path / "leave.md"
    source.write_text(
        "# Leave\n\n## Rule\n\nEmployees receive 20 days.", encoding="utf-8"
    )
    store = DocumentStore(tmp_path / "state.sqlite")
    initial = store.ingest(
        source,
        document_id="HR-1",
        title="Leave",
        owner="People",
        allowed_roles=["employee"],
    )
    old_chunk = store.current_chunks("HR-1")[0]
    assert initial.version == 1
    assert (
        store.ingest(
            source,
            document_id="HR-1",
            title="Leave",
            owner="People",
            allowed_roles=["employee"],
        ).version
        == 1
    )

    source.write_text(
        "# Leave\n\n## Rule\n\nEmployees receive 22 days.", encoding="utf-8"
    )
    updated = store.ingest(
        source,
        document_id="HR-1",
        title="Leave",
        owner="People",
        allowed_roles=["employee"],
    )
    assert updated.version == 2
    assert updated.content_hash != initial.content_hash
    assert old_chunk.chunk_id not in {c.chunk_id for c in store.current_chunks("HR-1")}
    assert "22 days" in " ".join(c.text for c in store.current_chunks("HR-1"))

    store.set_allowed_roles("HR-1", ["hr_admin"])
    assert store.get_document("HR-1").allowed_roles == ("hr_admin",)
    store.archive("HR-1")
    assert store.current_chunks("HR-1") == []
    assert not store.get_document("HR-1").active
    store.ingest(
        source,
        document_id="HR-1",
        title="Leave",
        owner="People",
        allowed_roles=["hr_admin"],
    )
    assert not store.get_document("HR-1").active


def test_missing_text_and_unsupported_files_fail(tmp_path: Path) -> None:
    blank_pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with blank_pdf.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="no extractable text"):
        read_sections(blank_pdf)
    with pytest.raises(ValueError, match="Unsupported"):
        read_sections(tmp_path / "data.exe")


def test_complete_demo_collection_ingests_and_questions_align(tmp_path: Path) -> None:
    root = get_project_root() / "data" / "demo"
    _, documents = load_collection(root / "manifest.yaml")
    cases = validate_questions(
        root / "question_cases.jsonl",
        {entry["document_id"] for entry in documents},
    )
    assert len(documents) == 45
    assert len(cases) == 170
    assert len({entry["source"] for entry in documents}) == 45
    assert Counter(case["scenario"] for case in cases) == {
        "direct": 45,
        "paraphrase": 40,
        "multiple_sources": 40,
        "near_miss": 25,
        "access_denied": 7,
        "document_update": 4,
        "permission_change": 4,
        "unsupported": 5,
    }
    assert all(
        case["source_document_ids"] for case in cases if case["scenario"] == "direct"
    )
    answerable = [
        case
        for case in cases
        if case["scenario"] in {"direct", "paraphrase", "multiple_sources", "near_miss"}
    ]
    assert len(answerable) == 150
    assert len({case["question"].strip().casefold() for case in answerable}) == 150
    assert all(
        len(case["source_document_ids"]) == 2
        for case in answerable
        if case["scenario"] == "multiple_sources"
    )
    roles = {
        "employee_demo": "employee",
        "hr_demo": "hr_admin",
        "it_admin_demo": "it_admin",
    }
    by_id = {entry["document_id"]: entry for entry in documents}
    assert all(
        roles[case["user_id"]] in by_id[document_id]["allowed_roles"]
        for case in answerable
        for document_id in case["source_document_ids"]
    )

    database = tmp_path / "demo.sqlite"
    assert ingest_collection(root / "manifest.yaml", database) == 45
    store = DocumentStore(database)
    assert len(store.list_documents()) == 45
    assert sum(len(store.current_chunks(d["document_id"])) for d in documents) > 45
    assert all(store.current_chunks(d["document_id"]) for d in documents)
    assert store.current_chunks("PEO-001")[0].page_number == 1
    assert "20 paid annual leave days" in " ".join(
        c.text for c in store.current_chunks("PEO-001")
    )

    revision_case = next(
        case
        for case in cases
        if case["scenario"] == "document_update"
        and case["state_change"]["document_id"] == "PEO-001"
    )
    revision_path = root / revision_case["state_change"]["replacement_source"]
    assert revision_path.exists()
    revised = store.ingest(
        revision_path,
        document_id="PEO-001",
        title="Annual Leave Policy",
        owner="People Operations",
        allowed_roles=["employee"],
    )
    assert revised.version == 2
    assert "22 paid annual leave days" in " ".join(
        c.text for c in store.current_chunks("PEO-001")
    )


def test_question_ids_and_sources_are_plain_json() -> None:
    path = get_project_root() / "data/demo/question_cases.jsonl"
    with path.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    assert [case["case_id"] for case in cases] == [
        f"Q{number:03d}" for number in range(1, 171)
    ]
