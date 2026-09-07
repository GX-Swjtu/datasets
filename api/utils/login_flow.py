"""Redirect validation and public results for browser sign-in."""

import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit


def safe_return_to(value):
    if not isinstance(value, str) or len(value) > 4096:
        return "/"
    decoded = value
    # Reject nested escaping as well as browser-normalized slash/dot paths.
    for _ in range(4):
        expanded = unquote(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    if "%" in decoded.split("?", 1)[0].split("#", 1)[0]:
        return "/"
    if not decoded.startswith("/") or decoded.startswith("//") or re.search(r"[\\\x00-\x1f\x7f]", decoded):
        return "/"
    parsed = urlsplit(decoded)
    segments = []
    for segment in parsed.path.split("/"):
        if segment == "..":
            if segments:
                segments.pop()
        elif segment and segment != ".":
            segments.append(segment)
    if segments and segments[0].lower() in {"login", "login-next", "api", "v1", "admin"}:
        return "/"
    # Authentication transport parameters must never become a return target.
    if any(key.lower() in {"auth", "code", "state", "error", "return_to"} for key, _ in parse_qsl(parsed.query)):
        return "/"
    return value


def login_result_url(return_to="/", *, auth=None, error=None):
    parameters = {"return_to": safe_return_to(return_to)}
    if auth:
        parameters["auth"] = auth
    if error:
        parameters["error"] = error
    return "/login?" + urlencode(parameters)
