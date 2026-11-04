"""Data ingestion, processing, and manifest management."""

from nool.data.hotpotqa import (
    CorpusRecord,
    HotpotQADataError,
    HotpotQAError,
    HotpotQAMissingMemberError,
    QrelRecord,
    QueryRecord,
    audit_hotpotqa_archive,
    compute_sha256,
    iter_corpus,
    iter_queries,
    read_qrels,
)
from nool.data.sampling import (
    DEFAULT_CORPUS_SAMPLE_SIZE,
    DEFAULT_DEV_QUERY_COUNT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_QUERY_COUNT,
    SampleResult,
    prepare_hotpotqa_sample,
)

__all__ = [
    "DEFAULT_CORPUS_SAMPLE_SIZE",
    "DEFAULT_DEV_QUERY_COUNT",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TEST_QUERY_COUNT",
    "CorpusRecord",
    "HotpotQADataError",
    "HotpotQAError",
    "HotpotQAMissingMemberError",
    "QrelRecord",
    "QueryRecord",
    "SampleResult",
    "audit_hotpotqa_archive",
    "compute_sha256",
    "iter_corpus",
    "iter_queries",
    "prepare_hotpotqa_sample",
    "read_qrels",
]
