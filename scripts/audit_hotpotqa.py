#!/usr/bin/env python3
"""Audit the HotpotQA raw archive and generate or update the dataset manifest."""

import argparse
import sys
from datetime import UTC, datetime

import yaml

from nool.config import get_project_root, resolve_path
from nool.data.hotpotqa import (
    HotpotQAError,
    audit_hotpotqa_archive,
)

HOTPOTQA_SOURCE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit HotpotQA raw dataset archive without full extraction."
    )
    parser.add_argument(
        "--archive",
        type=str,
        default="data/raw/hotpotqa.zip",
        help="Path to the HotpotQA zip archive (default: data/raw/hotpotqa.zip).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/hotpotqa.yaml",
        help="Path for updated manifest (default: data/manifests/hotpotqa.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the dataset audit."""
    args = parse_args()
    archive_path = resolve_path(args.archive)
    manifest_path = resolve_path(args.manifest)

    if not archive_path.exists():
        sys.stderr.write(f"Error: Archive not found at {archive_path}\n")
        sys.stderr.write(f"If missing, download from: {HOTPOTQA_SOURCE_URL}\n")
        sys.exit(1)

    print(f"Auditing archive: {archive_path}")
    try:
        audit_results = audit_hotpotqa_archive(archive_path)
    except HotpotQAError as exc:
        sys.stderr.write(f"Audit failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected error during audit: {exc}\n")
        sys.exit(1)

    # Relative path for manifest recording if inside project root
    root = get_project_root()
    try:
        archive_display_path = str(archive_path.relative_to(root)).replace("\\", "/")
    except ValueError:
        archive_display_path = str(archive_path)

    inspection_date = datetime.now(UTC).strftime("%Y-%m-%d")

    manifest_data = {
        "dataset": {
            "name": "HotpotQA",
            "source_url": HOTPOTQA_SOURCE_URL,
            "archive_file": archive_display_path,
            "inspection_date": inspection_date,
            "file_size_bytes": audit_results["file_size_bytes"],
            "sha256": audit_results["sha256"],
            "is_valid_zip": audit_results["is_valid_zip"],
            "archive_contents": audit_results["archive_contents"],
            "record_counts": audit_results["record_counts"],
            "data_quality": audit_results["data_quality"],
            "audit_status": audit_results["audit_status"],
        }
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    print(f"Audit completed successfully. Manifest written to: {manifest_path}")
    print(f"Archive Size: {audit_results['file_size_bytes']} bytes")
    print(f"SHA-256: {audit_results['sha256']}")
    print(f"Corpus records: {audit_results['record_counts']['corpus']}")
    print(f"Query records: {audit_results['record_counts']['queries']}")
    print(f"Qrels records: {audit_results['record_counts']['qrels']}")
    print(f"Data quality: {audit_results['data_quality']}")


if __name__ == "__main__":
    main()
