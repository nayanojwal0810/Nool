"""Unit tests for HotpotQA dataset reading, validation, and sampling."""

import json
import zipfile
from pathlib import Path

import pytest

from nool.data.hotpotqa import (
    HotpotQADataError,
    HotpotQAError,
    HotpotQAMissingMemberError,
    audit_hotpotqa_archive,
    compute_sha256,
    iter_corpus,
    iter_queries,
    read_qrels,
)
from nool.data.sampling import prepare_hotpotqa_sample


def _create_mock_zip(
    zip_path: Path,
    corpus_records: list[dict] | None = None,
    query_records: list[dict] | None = None,
    dev_qrels_text: str | None = None,
    test_qrels_text: str | None = None,
    custom_files: dict[str, str] | None = None,
) -> Path:
    """Helper creating a test zip archive with specified content."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        if corpus_records is not None:
            content = "\n".join(json.dumps(r) for r in corpus_records) + "\n"
            zf.writestr("hotpotqa/corpus.jsonl", content)

        if query_records is not None:
            content = "\n".join(json.dumps(r) for r in query_records) + "\n"
            zf.writestr("hotpotqa/queries.jsonl", content)

        if dev_qrels_text is not None:
            zf.writestr("hotpotqa/qrels/dev.tsv", dev_qrels_text)

        if test_qrels_text is not None:
            zf.writestr("hotpotqa/qrels/test.tsv", test_qrels_text)

        if custom_files:
            for fname, fcontent in custom_files.items():
                zf.writestr(fname, fcontent)

    return zip_path


def test_read_valid_corpus_and_queries(tmp_path: Path) -> None:
    """Verify reading well-formed corpus and query records."""
    corpus = [
        {"_id": "c1", "title": "Doc 1", "text": "Content 1", "metadata": {"k": "v"}},
        {"_id": "c2", "text": "Content 2 without title"},
    ]
    queries = [
        {"_id": "q1", "text": "What is doc 1?"},
        {"_id": "q2", "text": "What is doc 2?", "metadata": {"answer": "Content 2"}},
    ]
    dev_qrels = "query-id\tcorpus-id\tscore\nq1\tc1\t1\n"
    test_qrels = "query-id\tcorpus-id\tscore\nq2\tc2\t1\n"

    zip_file = tmp_path / "valid.zip"
    _create_mock_zip(zip_file, corpus, queries, dev_qrels, test_qrels)

    docs = list(iter_corpus(zip_file))
    assert len(docs) == 2
    assert docs[0].id == "c1"
    assert docs[0].title == "Doc 1"
    assert docs[0].text == "Content 1"
    assert docs[0].metadata == {"k": "v"}
    assert docs[1].id == "c2"
    assert docs[1].title == ""

    qs = list(iter_queries(zip_file))
    assert len(qs) == 2
    assert qs[0].id == "q1"
    assert qs[0].text == "What is doc 1?"
    assert qs[1].metadata == {"answer": "Content 2"}


def test_read_valid_qrels(tmp_path: Path) -> None:
    """Verify reading dev and test relevance judgments."""
    dev_qrels = "query-id\tcorpus-id\tscore\nq1\tc1\t1\nq1\tc2\t1\n"
    test_qrels = "query-id\tcorpus-id\tscore\nq2\tc3\t1\n"

    zip_file = tmp_path / "qrels.zip"
    _create_mock_zip(zip_file, [], [], dev_qrels, test_qrels)

    dev_records = read_qrels(zip_file, split="dev")
    assert len(dev_records) == 2
    assert dev_records[0].query_id == "q1"
    assert dev_records[0].corpus_id == "c1"
    assert dev_records[0].score == 1

    test_records = read_qrels(zip_file, split="test")
    assert len(test_records) == 1
    assert test_records[0].query_id == "q2"


def test_reject_malformed_jsonl(tmp_path: Path) -> None:
    """Verify that syntax errors and missing required fields raise HotpotQADataError."""
    # Syntax error
    zip_syntax = tmp_path / "bad_syntax.zip"
    with zipfile.ZipFile(zip_syntax, "w") as zf:
        zf.writestr("hotpotqa/corpus.jsonl", '{"_id": "c1", "text": "ok"}\n{invalid\n')

    with pytest.raises(HotpotQADataError) as exc_info:
        list(iter_corpus(zip_syntax))
    assert exc_info.value.line_number == 2

    # Missing _id field
    zip_missing_id = tmp_path / "missing_id.zip"
    with zipfile.ZipFile(zip_missing_id, "w") as zf:
        zf.writestr("hotpotqa/corpus.jsonl", '{"text": "missing id"}\n')

    with pytest.raises(HotpotQADataError) as exc_info2:
        list(iter_corpus(zip_missing_id))
    assert exc_info2.value.line_number == 1

    # Missing text field in queries
    zip_missing_text = tmp_path / "missing_text.zip"
    with zipfile.ZipFile(zip_missing_text, "w") as zf:
        zf.writestr("hotpotqa/queries.jsonl", '{"_id": "q1"}\n')

    with pytest.raises(HotpotQADataError) as exc_info3:
        list(iter_queries(zip_missing_text))
    assert exc_info3.value.line_number == 1


def test_reject_malformed_qrels(tmp_path: Path) -> None:
    """Verify that invalid headers or values in TSV raise HotpotQADataError."""
    # Invalid header
    zip_header = tmp_path / "bad_header.zip"
    _create_mock_zip(zip_header, dev_qrels_text="qid\tcid\tscore\nq1\tc1\t1\n")
    with pytest.raises(HotpotQADataError) as exc:
        read_qrels(zip_header, split="dev")
    assert "Invalid header" in str(exc.value)

    # Non-integer score
    zip_score = tmp_path / "bad_score.zip"
    _create_mock_zip(
        zip_score, dev_qrels_text="query-id\tcorpus-id\tscore\nq1\tc1\thigh\n"
    )
    with pytest.raises(HotpotQADataError) as exc2:
        read_qrels(zip_score, split="dev")
    assert exc2.value.line_number == 2
    assert "Invalid integer score" in str(exc2.value)


def test_handling_missing_archive_members(tmp_path: Path) -> None:
    """Verify that missing required files raise HotpotQAMissingMemberError."""
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("unrelated.txt", "hello")

    with pytest.raises(HotpotQAMissingMemberError):
        list(iter_corpus(empty_zip))

    with pytest.raises(HotpotQAMissingMemberError):
        list(iter_queries(empty_zip))

    with pytest.raises(HotpotQAMissingMemberError):
        read_qrels(empty_zip, split="dev")


def test_reject_invalid_zip_file(tmp_path: Path) -> None:
    """Verify that non-zip files or corrupted archives are rejected."""
    bad_file = tmp_path / "not_a_zip.zip"
    bad_file.write_text("plain text, not a zip", encoding="utf-8")

    with pytest.raises(HotpotQAError) as exc:
        list(iter_corpus(bad_file))
    assert "Not a valid ZIP archive" in str(exc.value)


def test_reproducible_sample_generation(tmp_path: Path) -> None:
    """Verify reproducible sampling, positive retention, and disjoint splits."""
    # 12 corpus passages: c1..c12
    corpus = [
        {"_id": f"c{i}", "title": f"T{i}", "text": f"Passage {i}"} for i in range(1, 13)
    ]
    # 4 dev queries, 4 test queries
    queries = [{"_id": f"q_dev_{i}", "text": f"Dev query {i}"} for i in range(1, 5)] + [
        {"_id": f"q_test_{i}", "text": f"Test query {i}"} for i in range(1, 5)
    ]
    # Dev qrels: q_dev_1 -> c1, c2; q_dev_2 -> c2, c3; q_dev_3 -> c4; q_dev_4 -> c5
    dev_qrels = (
        "query-id\tcorpus-id\tscore\n"
        "q_dev_1\tc1\t1\n"
        "q_dev_1\tc2\t1\n"
        "q_dev_2\tc2\t1\n"
        "q_dev_2\tc3\t1\n"
        "q_dev_3\tc4\t1\n"
        "q_dev_4\tc5\t1\n"
    )
    # Test qrels: q_test_1 -> c6; q_test_2 -> c7; q_test_3 -> c8; q_test_4 -> c9
    test_qrels = (
        "query-id\tcorpus-id\tscore\n"
        "q_test_1\tc6\t1\n"
        "q_test_2\tc7\t1\n"
        "q_test_3\tc8\t1\n"
        "q_test_4\tc9\t1\n"
    )

    zip_file = tmp_path / "sample_source.zip"
    _create_mock_zip(zip_file, corpus, queries, dev_qrels, test_qrels)

    out_1 = tmp_path / "out_1"
    out_2 = tmp_path / "out_2"

    # Sample: target size 7, 2 dev queries, 2 test queries, seed 42
    res1 = prepare_hotpotqa_sample(
        archive_path=zip_file,
        output_dir=out_1,
        corpus_sample_size=7,
        dev_query_count=2,
        test_query_count=2,
        seed=42,
    )
    res2 = prepare_hotpotqa_sample(
        archive_path=zip_file,
        output_dir=out_2,
        corpus_sample_size=7,
        dev_query_count=2,
        test_query_count=2,
        seed=42,
    )

    assert res1.corpus_sample_size == 7
    assert res1.dev_query_count == 2
    assert res1.test_query_count == 2
    assert res1.positive_passage_count + res1.negative_passage_count == 7
    assert res2.corpus_sample_size == res1.corpus_sample_size
    assert res2.positive_passage_count == res1.positive_passage_count

    # Exact bit-for-bit reproducibility check
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels/dev.tsv", "qrels/test.tsv"]:
        content1 = (out_1 / fname).read_text(encoding="utf-8")
        content2 = (out_2 / fname).read_text(encoding="utf-8")
        assert content1 == content2, f"Mismatch in reproducible file {fname}"

    # Verify that all selected positive passages are included in the sampled corpus
    corpus_lines = [
        json.loads(line)
        for line in (out_1 / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    sampled_cids = {doc["_id"] for doc in corpus_lines}
    assert len(sampled_cids) == 7

    for split in ["dev.tsv", "test.tsv"]:
        lines = (out_1 / "qrels" / split).read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            if not line:
                continue
            qid, cid, score = line.split("\t")
            assert cid in sampled_cids, (
                f"Positive passage {cid} for {qid} missing from corpus!"
            )


def test_sampling_positive_exceeds_sample_size_raises(tmp_path: Path) -> None:
    """Verify error when required positive passages exceed sample size."""
    corpus = [{"_id": f"c{i}", "text": f"Text {i}"} for i in range(1, 6)]
    queries = [{"_id": "q1", "text": "Q1"}, {"_id": "q2", "text": "Q2"}]
    dev_qrels = "query-id\tcorpus-id\tscore\nq1\tc1\t1\nq1\tc2\t1\nq1\tc3\t1\n"
    test_qrels = "query-id\tcorpus-id\tscore\nq2\tc4\t1\nq2\tc5\t1\n"

    zip_file = tmp_path / "overflow.zip"
    _create_mock_zip(zip_file, corpus, queries, dev_qrels, test_qrels)

    with pytest.raises(ValueError) as exc:
        # Sample size 3 is smaller than 5 required positive passages
        prepare_hotpotqa_sample(
            archive_path=zip_file,
            output_dir=tmp_path / "out",
            corpus_sample_size=3,
            dev_query_count=1,
            test_query_count=1,
            seed=42,
        )
    assert "exceed configured corpus sample size" in str(exc.value)


def test_audit_archive_on_mock(tmp_path: Path) -> None:
    """Verify that audit_hotpotqa_archive produces accurate records on mock zip."""
    corpus = [{"_id": f"c{i}", "text": f"P{i}"} for i in range(1, 6)]
    queries = [{"_id": "q1", "text": "Q1"}, {"_id": "q2", "text": "Q2"}]
    dev_qrels = "query-id\tcorpus-id\tscore\nq1\tc1\t1\n"
    test_qrels = "query-id\tcorpus-id\tscore\nq2\tc2\t1\n"

    zip_file = tmp_path / "mock_audit.zip"
    _create_mock_zip(zip_file, corpus, queries, dev_qrels, test_qrels)

    report = audit_hotpotqa_archive(zip_file)
    assert report["is_valid_zip"] is True
    assert report["record_counts"]["corpus"] == 5
    assert report["record_counts"]["queries"] == 2
    assert report["record_counts"]["qrels"]["dev"] == 1
    assert report["record_counts"]["qrels"]["test"] == 1
    assert report["data_quality"]["missing_or_malformed_records"] == 0
    assert report["data_quality"]["qrels_queries_missing_from_queries"] == 0
    assert report["data_quality"]["qrels_corpus_missing_from_corpus"] == 0
    assert report["sha256"] == compute_sha256(zip_file)
