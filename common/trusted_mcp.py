"""Deployment-owned MCP destinations allowed to receive BFF identity context."""
import os
import json

CONTEXT_HEADERS = frozenset({
    "x-ngl-user-ldap", "x-ngl-application", "x-ngl-workflow-id",
    "x-ngl-correlation-id", "x-ngl-conversation-id", "x-ngl-turn-id", "x-ngl-message-id",
})


def authenticated_context_headers(custom, auth_type, workflow_id):
    """A browser login cannot supply a BFF actor, even on an accessible canvas."""
    if isinstance(custom, str):
        try:
            custom = json.loads(custom)
        except (ValueError, TypeError):
            custom = {}
    if not isinstance(custom, dict):
        return {}
    normalized = {str(key).lower(): value for key, value in custom.items()}
    trusted = auth_type == "API" and normalized.get("x-ngl-workflow-id") == workflow_id
    return {key: value for key, value in custom.items()
            if not str(key).lower().startswith("x-ngl-")
            or (trusted and str(key).lower() in CONTEXT_HEADERS)}


def trusted_internal_urls():
    return {value for key in ("MCP_TRUSTED_INTERNAL_URL", "MCP_OA_INTERNAL_URL")
            if (value := os.environ.get(key, ""))}


def bound_context_headers(url, configured, custom):
    # Configured headers can never impersonate a BFF actor. Browser/Begin values
    # cannot override service credentials, nor can context leak to a public MCP.
    headers = {k: v for k, v in configured.items() if not k.lower().startswith("x-ngl-")}
    trusted = url in trusted_internal_urls()
    for key, value in custom.items():
        lower = key.lower()
        if lower.startswith("x-ngl-"):
            if trusted and lower in CONTEXT_HEADERS:
                headers[key] = value
        elif not trusted:
            headers[key] = value
    return headers
