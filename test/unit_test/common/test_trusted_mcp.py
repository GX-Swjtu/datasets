from common.trusted_mcp import authenticated_context_headers, bound_context_headers


def test_only_exact_destinations_receive_context(monkeypatch):
    url = "http://auth-oa-mcp.dev.svc.cluster.local:8000/mcp"
    monkeypatch.setenv("MCP_OA_INTERNAL_URL", url)
    static = {"Authorization": "secret", "X-NGL-User-Ldap": "spoofed"}
    custom = {"X-NGL-User-Ldap": "actor", "Authorization": "override", "X-NGL-Unknown": "x"}
    assert bound_context_headers(url, static, custom) == {
        "Authorization": "secret", "X-NGL-User-Ldap": "actor",
    }
    for other in (url + "/", "https://public.invalid/mcp"):
        assert not any(k.lower().startswith("x-ngl-")
                       for k in bound_context_headers(other, static, custom))


def test_browser_and_wrong_workflow_cannot_impersonate_bff():
    headers = {"X-NGL-User-Ldap": "actor", "X-NGL-Workflow-Id": "flow", "Other": "value"}
    for auth_type, flow in (("JWT", "flow"), ("API", "other"), (None, "flow")):
        assert authenticated_context_headers(headers, auth_type, flow) == {"Other": "value"}
    assert authenticated_context_headers(headers, "API", "flow") == headers
    assert authenticated_context_headers('{"X-NGL-User-Ldap":"spoof"}', "JWT", "flow") == {}
