"""Map signed-in people to the demo collection's access roles."""

from collections.abc import Mapping

from nool.retrieval.company import DEMO_USERS


def demo_user_for_email(
    email: str | None, assignments: Mapping[str, str]
) -> str | None:
    """Return only a known role assigned to the signed-in email address."""
    if not isinstance(email, str) or not email.strip():
        return None
    normalized = email.strip().casefold()
    for address, user_id in assignments.items():
        if address.strip().casefold() == normalized:
            return user_id if user_id in DEMO_USERS else None
    return None


def citation_is_current(citation, user_id: str, service) -> bool:
    """Recheck a displayed source against the viewer's current role."""
    document = service.store.get_document(citation.document_id)
    return bool(
        document is not None
        and document.active
        and document.version == citation.version
        and DEMO_USERS[user_id] in document.allowed_roles
    )
