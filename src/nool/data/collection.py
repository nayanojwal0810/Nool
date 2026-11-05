"""Validate and ingest the fictional company document collection."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from nool.data.documents import DocumentStore, read_sections

KNOWN_ROLES = {"employee", "hr_admin", "it_admin"}
KNOWN_SCENARIOS = {
    "direct",
    "paraphrase",
    "multiple_sources",
    "near_miss",
    "access_denied",
    "document_update",
    "permission_change",
    "unsupported",
}


@dataclass(frozen=True)
class CollectionSyncResult:
    created: tuple[str, ...]
    content_updated: tuple[str, ...]
    access_changed: tuple[str, ...]
    archived: tuple[str, ...]
    reactivated: tuple[str, ...]
    title_updated: tuple[str, ...]

    @property
    def requires_rebuild(self) -> bool:
        return bool(
            self.created
            or self.content_updated
            or self.archived
            or self.reactivated
            or self.title_updated
        )


def load_collection(manifest_path: Path | str) -> tuple[str, list[dict]]:
    """Read the document list and reject missing, duplicate, or unsafe sources."""
    path = Path(manifest_path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("company"), str):
        raise ValueError("Collection manifest needs a company name")
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Collection manifest needs documents")
    seen_ids: set[str] = set()
    seen_sources: set[Path] = set()
    for entry in documents:
        if not isinstance(entry, dict):
            raise ValueError("Each document entry must be a mapping")
        for key in ("document_id", "title", "source", "owner", "allowed_roles"):
            if not entry.get(key):
                raise ValueError(f"Document entry is missing {key}")
        document_id = entry["document_id"]
        source = (path.parent / entry["source"]).resolve()
        if document_id in seen_ids or source in seen_sources:
            raise ValueError(f"Duplicate document ID or source: {document_id}")
        if not source.is_relative_to(path.parent) or not source.is_file():
            raise ValueError(
                f"Document source is missing or outside collection: {source}"
            )
        roles = entry["allowed_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) for role in roles)
            or not set(roles) <= KNOWN_ROLES
        ):
            raise ValueError(f"Invalid allowed roles for {document_id}")
        if entry.get("version") != 1 or not entry.get("effective_date"):
            raise ValueError(f"Invalid starting version or date for {document_id}")
        source_text = "\n".join(section.text for section in read_sections(source))
        if document_id not in source_text or entry["title"] not in source_text:
            raise ValueError(f"Source identity does not match manifest: {document_id}")
        seen_ids.add(document_id)
        seen_sources.add(source)
    return data["company"], documents


def validate_questions(question_path: Path | str, document_ids: set[str]) -> list[dict]:
    """Check case IDs, source references, and expected outcomes."""
    cases = []
    seen: set[str] = set()
    with Path(question_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            case = json.loads(line)
            case_id = case.get("case_id")
            if not case_id or case_id in seen:
                raise ValueError(f"Missing or duplicate case ID at line {number}")
            if not case.get("question") or not case.get("expected_answer"):
                raise ValueError(f"Incomplete question case: {case_id}")
            if case.get("scenario") not in KNOWN_SCENARIOS:
                raise ValueError(f"Unknown scenario: {case_id}")
            if case.get("user_id") not in {"employee_demo", "hr_demo", "it_admin_demo"}:
                raise ValueError(f"Unknown demo user: {case_id}")
            sources = case.get("source_document_ids")
            if not isinstance(sources, list) or not set(sources) <= document_ids:
                raise ValueError(f"Unknown source document in case: {case_id}")
            change = case.get("state_change")
            if change is not None and not isinstance(change, dict):
                raise ValueError(f"Invalid state change: {case_id}")
            if change and change.get("document_id") not in document_ids:
                raise ValueError(f"Unknown changed document in case: {case_id}")
            if case["scenario"] == "document_update":
                if not change or not isinstance(change.get("replacement_source"), str):
                    raise ValueError(f"Missing replacement source: {case_id}")
                replacement = (
                    Path(question_path).parent / change["replacement_source"]
                ).resolve()
                if not replacement.is_relative_to(Path(question_path).resolve().parent):
                    raise ValueError(f"Unsafe replacement path: {case_id}")
                if not replacement.is_file() or change.get("expected_version") != 2:
                    raise ValueError(f"Invalid replacement source: {case_id}")
            if (
                case["scenario"]
                in {"access_denied", "permission_change", "unsupported"}
                and sources
            ):
                raise ValueError(
                    f"Denied or unsupported case exposes a source: {case_id}"
                )
            seen.add(case_id)
            cases.append(case)
    return cases


def ingest_collection(manifest_path: Path | str, database_path: Path | str) -> int:
    """Ingest every source in the manifest and return its document count."""
    _, documents = load_collection(manifest_path)
    root = Path(manifest_path).resolve().parent
    store = DocumentStore(database_path)
    for entry in documents:
        store.ingest(
            root / entry["source"],
            document_id=entry["document_id"],
            title=entry["title"],
            owner=entry["owner"],
            allowed_roles=entry["allowed_roles"],
        )
    return len(documents)


def sync_collection(
    manifest_path: Path | str, database_path: Path | str
) -> CollectionSyncResult:
    """Apply source, access, and removal changes from a validated manifest."""
    _, entries = load_collection(manifest_path)
    root = Path(manifest_path).resolve().parent
    store = DocumentStore(database_path)
    before = {doc.document_id: doc for doc in store.list_documents()}
    current_ids = {entry["document_id"] for entry in entries}
    created: list[str] = []
    content_updated: list[str] = []
    access_changed: list[str] = []
    reactivated: list[str] = []
    title_updated: list[str] = []

    for entry in entries:
        document_id = entry["document_id"]
        previous = before.get(document_id)
        roles = tuple(sorted(entry["allowed_roles"]))
        if previous and previous.allowed_roles != roles:
            store.set_allowed_roles(document_id, list(roles))
            access_changed.append(document_id)
        current = store.ingest(
            root / entry["source"],
            document_id=document_id,
            title=entry["title"],
            owner=entry["owner"],
            allowed_roles=entry["allowed_roles"],
        )
        if previous is None:
            created.append(document_id)
        else:
            if current.version != previous.version:
                content_updated.append(document_id)
            if current.title != previous.title:
                title_updated.append(document_id)
            if not previous.active:
                store.activate(document_id)
                reactivated.append(document_id)

    archived = []
    for document_id, previous in before.items():
        if previous.active and document_id not in current_ids:
            store.archive(document_id)
            archived.append(document_id)
    return CollectionSyncResult(
        tuple(created),
        tuple(content_updated),
        tuple(access_changed),
        tuple(archived),
        tuple(reactivated),
        tuple(title_updated),
    )
