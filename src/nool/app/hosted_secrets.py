"""Write Streamlit authentication settings from host-provided secrets."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from nool.retrieval.company import DEMO_USERS

AUTH_VARIABLES = {
    "redirect_uri": "NOOL_AUTH_REDIRECT_URI",
    "cookie_secret": "NOOL_AUTH_COOKIE_SECRET",
    "client_id": "NOOL_AUTH_CLIENT_ID",
    "client_secret": "NOOL_AUTH_CLIENT_SECRET",
    "server_metadata_url": "NOOL_AUTH_SERVER_METADATA_URL",
}


def write_hosted_secrets(destination: Path, environment: Mapping[str, str]) -> None:
    """Create a private TOML file for OIDC and assigned demo roles."""
    auth = {}
    for field, variable in AUTH_VARIABLES.items():
        value = environment.get(variable, "").strip()
        if not value:
            raise ValueError(f"Missing hosted setting: {variable}")
        auth[field] = value
    if len(auth["cookie_secret"]) < 32:
        raise ValueError("Cookie secret must have at least 32 characters")
    for field in ("redirect_uri", "server_metadata_url"):
        parsed = urlparse(auth[field])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username:
            raise ValueError(f"{field} must be a secure URL")
    assignments = json.loads(environment.get("NOOL_ACCESS_JSON", "{}"))
    if (
        not isinstance(assignments, dict)
        or not assignments
        or any(
            not isinstance(email, str)
            or not isinstance(user_id, str)
            or "@" not in email
            or user_id not in DEMO_USERS
            for email, user_id in assignments.items()
        )
    ):
        raise ValueError("NOOL_ACCESS_JSON needs email-to-role assignments")
    normalized_emails = [email.strip().casefold() for email in assignments]
    if len(set(normalized_emails)) != len(normalized_emails):
        raise ValueError("NOOL_ACCESS_JSON has duplicate email assignments")
    lines = ["[auth]"]
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in auth.items())
    lines.append("")
    lines.append("[access]")
    lines.extend(
        f"{json.dumps(email.strip().casefold())} = {json.dumps(user_id)}"
        for email, user_id in assignments.items()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    destination.chmod(0o600)


if __name__ == "__main__":
    write_hosted_secrets(Path.home() / ".streamlit/secrets.toml", os.environ)
