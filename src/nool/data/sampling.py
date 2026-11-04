"""Reproducible sampling of HotpotQA queries and corpus passages."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nool.data.hotpotqa import (
    CorpusRecord,
    QrelRecord,
    QueryRecord,
    iter_corpus,
    iter_queries,
    read_qrels,
)

DEFAULT_CORPUS_SAMPLE_SIZE = 50_000
DEFAULT_DEV_QUERY_COUNT = 500
DEFAULT_TEST_QUERY_COUNT = 500
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class SampleResult:
    """Metadata summary of a generated HotpotQA sample."""

    corpus_sample_size: int
    dev_query_count: int
    test_query_count: int
    random_seed: int
    positive_passage_count: int
    negative_passage_count: int
    dev_qrel_count: int
    test_qrel_count: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Convert sample result metadata to a dictionary."""
        return {
            "corpus_sample_size": self.corpus_sample_size,
            "dev_query_count": self.dev_query_count,
            "test_query_count": self.test_query_count,
            "random_seed": self.random_seed,
            "positive_passage_count": self.positive_passage_count,
            "negative_passage_count": self.negative_passage_count,
            "dev_qrel_count": self.dev_qrel_count,
            "test_qrel_count": self.test_qrel_count,
            "output_dir": self.output_dir,
        }


def prepare_hotpotqa_sample(
    archive_path: Path | str,
    output_dir: Path | str,
    corpus_sample_size: int = DEFAULT_CORPUS_SAMPLE_SIZE,
    dev_query_count: int = DEFAULT_DEV_QUERY_COUNT,
    test_query_count: int = DEFAULT_TEST_QUERY_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> SampleResult:
    """Extract a reproducible sample of queries, qrels, and passages from HotpotQA.

    The sample retains all positive passages for selected development and test queries,
    filling the remaining slots with reproducibly sampled negative passages via
    streaming reservoir sampling.

    Args:
        archive_path: Path to the raw hotpotqa.zip archive.
        output_dir: Directory where sample files will be written.
        corpus_sample_size: Total number of passages in the final sample.
        dev_query_count: Number of development queries to select.
        test_query_count: Number of test queries to select.
        seed: Random seed for deterministic sampling.

    Returns:
        SampleResult with detailed sampling statistics.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If constraints are violated (e.g. insufficient passages,
            missing positive passages, or positive count exceeds sample size).
    """
    archive = Path(archive_path)
    out_path = Path(output_dir)
    rng = random.Random(seed)

    # 1. Load all queries into memory
    queries_by_id: dict[str, QueryRecord] = {q.id: q for q in iter_queries(archive)}

    # 2. Read qrels for dev and test splits
    dev_qrels = read_qrels(archive, split="dev")
    test_qrels = read_qrels(archive, split="test")

    # 3. Eligible query IDs (must have both query record and qrels)
    eligible_dev_qids = sorted(
        {q.query_id for q in dev_qrels if q.query_id in queries_by_id}
    )
    eligible_test_qids = sorted(
        {q.query_id for q in test_qrels if q.query_id in queries_by_id}
    )

    dev_k = min(dev_query_count, len(eligible_dev_qids))
    test_k = min(test_query_count, len(eligible_test_qids))

    selected_dev_qids = sorted(rng.sample(eligible_dev_qids, dev_k))
    selected_test_qids = sorted(rng.sample(eligible_test_qids, test_k))

    # Ensure dev and test queries are completely disjoint
    overlap = set(selected_dev_qids) & set(selected_test_qids)
    if overlap:
        raise ValueError(f"Development and test query selection overlap: {overlap}")

    selected_dev_set = set(selected_dev_qids)
    selected_test_set = set(selected_test_qids)

    # 4. Retain all qrels for selected queries
    selected_dev_qrels = [q for q in dev_qrels if q.query_id in selected_dev_set]
    selected_test_qrels = [q for q in test_qrels if q.query_id in selected_test_set]

    # 5. Identify all positive passage IDs
    required_positive_cids = {q.corpus_id for q in selected_dev_qrels} | {
        q.corpus_id for q in selected_test_qrels
    }

    if len(required_positive_cids) > corpus_sample_size:
        raise ValueError(
            f"Required positive passages ({len(required_positive_cids)}) "
            f"exceed configured corpus sample size ({corpus_sample_size})."
        )

    needed_negatives = corpus_sample_size - len(required_positive_cids)

    # 6. Stream corpus with reservoir sampling for negatives
    positive_records: dict[str, CorpusRecord] = {}
    negative_reservoir: list[CorpusRecord] = []
    negative_seen_count = 0

    for doc in iter_corpus(archive):
        if doc.id in required_positive_cids:
            positive_records[doc.id] = doc
        else:
            negative_seen_count += 1
            if len(negative_reservoir) < needed_negatives:
                negative_reservoir.append(doc)
            else:
                j = rng.randint(0, negative_seen_count - 1)
                if j < needed_negatives:
                    negative_reservoir[j] = doc

    # 7. Validate positive passage completeness
    missing_positives = required_positive_cids - set(positive_records.keys())
    if missing_positives:
        raise ValueError(
            f"Qrels reference {len(missing_positives)} passage(s) missing from corpus. "
            f"Missing IDs: {sorted(missing_positives)[:10]}"
        )

    total_available = len(positive_records) + negative_seen_count
    if total_available < corpus_sample_size:
        raise ValueError(
            f"Insufficient corpus records ({total_available}) for requested "
            f"sample size ({corpus_sample_size})."
        )

    # 8. Deterministically sort passages and queries for reproducible file output
    final_passages = sorted(
        list(positive_records.values()) + negative_reservoir,
        key=lambda r: r.id,
    )
    if len(final_passages) != corpus_sample_size:
        raise ValueError(
            f"Sample construction error: expected {corpus_sample_size} passages, "
            f"got {len(final_passages)}."
        )

    selected_query_records = sorted(
        [queries_by_id[qid] for qid in (selected_dev_set | selected_test_set)],
        key=lambda q: q.id,
    )

    sorted_dev_qrels = sorted(
        selected_dev_qrels,
        key=lambda q: (q.query_id, q.corpus_id),
    )
    sorted_test_qrels = sorted(
        selected_test_qrels,
        key=lambda q: (q.query_id, q.corpus_id),
    )

    # 9. Write sample files
    out_path.mkdir(parents=True, exist_ok=True)
    qrels_dir = out_path / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = out_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8", newline="\n") as f:
        for passage in final_passages:
            f.write(json.dumps(passage.to_dict()) + "\n")

    queries_file = out_path / "queries.jsonl"
    with open(queries_file, "w", encoding="utf-8", newline="\n") as f:
        for query in selected_query_records:
            f.write(json.dumps(query.to_dict()) + "\n")

    def write_tsv(target: Path, qrel_list: list[QrelRecord]) -> None:
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["query-id", "corpus-id", "score"])
            for qrel in qrel_list:
                writer.writerow([qrel.query_id, qrel.corpus_id, qrel.score])

    write_tsv(qrels_dir / "dev.tsv", sorted_dev_qrels)
    write_tsv(qrels_dir / "test.tsv", sorted_test_qrels)

    return SampleResult(
        corpus_sample_size=len(final_passages),
        dev_query_count=len(selected_dev_qids),
        test_query_count=len(selected_test_qids),
        random_seed=seed,
        positive_passage_count=len(positive_records),
        negative_passage_count=len(negative_reservoir),
        dev_qrel_count=len(sorted_dev_qrels),
        test_qrel_count=len(sorted_test_qrels),
        output_dir=str(out_path),
    )
