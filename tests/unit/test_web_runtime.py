"""Check hosted demo startup and signed-in role assignment."""

import zipfile
from pathlib import Path

import pytest

from nool.app.access import citation_is_current, demo_user_for_email
from nool.app.hosted_secrets import write_hosted_secrets
from nool.app.runtime import _extract_index, load_answer_service
from nool.generation.answers import Citation


def test_signed_in_email_needs_an_explicit_known_role() -> None:
    assignments = {
        "person@example.com": "employee_demo",
        "unknown@example.com": "administrator",
    }
    assert demo_user_for_email(" PERSON@EXAMPLE.COM ", assignments) == "employee_demo"
    assert demo_user_for_email("other@example.com", assignments) is None
    assert demo_user_for_email("unknown@example.com", assignments) is None
    assert demo_user_for_email(None, assignments) is None


def test_archive_rejects_paths_outside_runtime_directory(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe path"):
        _extract_index(archive_path, tmp_path / "runtime")
    assert not (tmp_path / "outside.txt").exists()


def test_published_index_matches_fresh_demo_ingestion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    service = load_answer_service(root, tmp_path)
    assert len(service.store.list_documents()) == 45
    assert service.search.indexed_fingerprint


def test_displayed_source_respects_current_permissions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    service = load_answer_service(root, tmp_path)
    document = service.store.get_document("PEO-002")
    chunk = service.store.current_chunks(document.document_id)[0]
    citation = Citation(
        document.document_id,
        document.title,
        chunk.chunk_id,
        document.version,
        chunk.section,
        chunk.page_number,
        chunk.text,
    )
    assert citation_is_current(citation, "employee_demo", service)
    service.store.set_allowed_roles(document.document_id, ["hr_admin"])
    assert not citation_is_current(citation, "employee_demo", service)


def test_hosted_settings_keep_roles_and_credentials_out_of_source(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "secrets.toml"
    environment = {
        "NOOL_AUTH_REDIRECT_URI": "https://nool.example/oauth2callback",
        "NOOL_AUTH_COOKIE_SECRET": "a" * 32,
        "NOOL_AUTH_CLIENT_ID": "client-id",
        "NOOL_AUTH_CLIENT_SECRET": "private-client-secret",
        "NOOL_AUTH_SERVER_METADATA_URL": "https://accounts.example/.well-known/openid-configuration",
        "NOOL_ACCESS_JSON": '{"Person@Example.com":"employee_demo"}',
    }
    write_hosted_secrets(destination, environment)
    settings = destination.read_text(encoding="utf-8")
    assert '"person@example.com" = "employee_demo"' in settings
    assert 'client_secret = "private-client-secret"' in settings
    with pytest.raises(ValueError, match="duplicate email"):
        write_hosted_secrets(
            destination,
            {
                **environment,
                "NOOL_ACCESS_JSON": (
                    '{"person@example.com":"employee_demo",'
                    '"PERSON@example.com":"hr_demo"}'
                ),
            },
        )
