"""Exercise browser login routes with real Quart requests and isolated services."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from quart import Blueprint, Quart

ROOT = Path(__file__).resolve().parents[5]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = load_file("browser_login_flow", ROOT / "api/utils/login_flow.py")


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.test",
        "//evil.test",
        "/%2fevil.test",
        "/%252fevil.test",
        "/\\evil.test",
        "/%0a/evil",
        "/login",
        "/login-next",
        "/api/v1/auth/logout",
        "/a/../login",
        "/v1/foo",
        "/admin/users",
        "/?auth=secret",
        "/?return_to=//evil.test",
    ],
)
def test_invalid_targets(target):
    assert flow.safe_return_to(target) == "/"


@pytest.mark.parametrize("target", ["/datasets?search=abc#list", "/datasets?search=two%20words#list", "/agent/123?tab=canvas#node-1"])
def test_deep_links(target):
    assert flow.safe_return_to(target) == target


@pytest.fixture
def login_app(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def noop(*args, **kwargs):
        return None

    def decorator(*args, **kwargs):
        return lambda function: function

    settings = SimpleNamespace(OAUTH_CONFIG={"ngl-auth": {"type": "oidc"}})
    stub("common", settings=settings)
    stub("common.time_utils", **dict.fromkeys(["current_timestamp", "datetime_format", "get_format_time"], noop))
    stub("common.misc_utils", download_img=AsyncMock(return_value=""), get_uuid=lambda: "random-state")
    stub("common.constants", RetCode=SimpleNamespace(SERVER_ERROR=500))
    stub("common.connection_utils", construct_response=noop)
    stub("api.db", FileType=SimpleNamespace(), UserTenantRole=SimpleNamespace())
    stub("api.db.services.file_service", FileService=SimpleNamespace())
    user = SimpleNamespace(email="developer@invalid.test", id="user-1", access_token="before", is_active="1", save=Mock(return_value=1), get_id=lambda: "signed-session")
    users = SimpleNamespace(query=Mock(return_value=[user]))
    stub("api.db.services.user_service", UserService=users, TenantService=SimpleNamespace(), UserTenantService=SimpleNamespace())
    stub("api.utils.api_utils", **dict.fromkeys(["get_data_error_result", "get_json_result", "get_request_json", "server_error_response"], noop), validate_request=decorator)
    stub("api.utils.nickname_validation", validate_nickname=noop)
    stub("api.utils.crypt", decrypt=noop)
    stub(
        "api.utils.web_utils",
        **dict.fromkeys(["send_email_html", "OTP_LENGTH", "OTP_TTL_SECONDS", "ATTEMPT_LIMIT", "ATTEMPT_LOCK_SECONDS", "RESEND_COOLDOWN_SECONDS", "otp_keys", "hash_code", "captcha_key"], noop),
    )
    monkeypatch.setitem(sys.modules, "api.utils.login_flow", flow)
    stub("rag.utils.redis_conn", REDIS_CONN=SimpleNamespace())
    logged_in = Mock()
    stub("api.apps", login_required=lambda function: function, current_user=user, login_user=logged_in, logout_user=noop)
    auth_client = SimpleNamespace(
        get_authorization_url=lambda state: f"https://auth.invalid/oauth/authorize?state={state}",
        async_exchange_code_for_token=AsyncMock(return_value={"access_token": "oidc-private", "id_token": "id-private"}),
        async_fetch_user_info=AsyncMock(return_value=SimpleNamespace(email=user.email, nickname="Developer", avatar_url="")),
    )
    client_factory = Mock(return_value=auth_client)
    stub("api.apps.auth", get_auth_client=client_factory)
    spec = importlib.util.spec_from_file_location("browser_login_user_api", ROOT / "api/apps/restful_apis/user_api.py")
    module = importlib.util.module_from_spec(spec)
    module.manager = Blueprint("login", __name__)
    spec.loader.exec_module(module)
    app = Quart(__name__)
    app.secret_key = "isolated-browser-login-test-key"
    app.register_blueprint(module.manager, url_prefix="/api/v1")
    return SimpleNamespace(client=app.test_client(), module=module, user=user, users=users, auth=auth_client, factory=client_factory, logged_in=logged_in)


async def start(fixture, target="/agent/123?tab=canvas#node"):
    response = await fixture.client.get("/api/v1/auth/login/ngl-auth", query_string={"return_to": target})
    assert response.status_code == 302
    return parse_qs(urlsplit(response.location).query)["state"][0]


def result(response):
    assert urlsplit(response.location).path == "/login"
    return parse_qs(urlsplit(response.location).query)


@pytest.mark.asyncio
async def test_existing_user_login_preserves_target_and_consumes_state(login_app):
    state = await start(login_app)
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"code": "code", "state": state})
    assert result(response) == {"auth": ["signed-session"], "return_to": ["/agent/123?tab=canvas#node"]}
    login_app.logged_in.assert_called_once_with(login_app.user)
    repeated = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"code": "code", "state": state})
    assert result(repeated)["error"] == ["invalid_state"]
    login_app.auth.async_exchange_code_for_token.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_login_keeps_native_registration(login_app, monkeypatch):
    login_app.users.query.return_value = []
    register = Mock(return_value=[login_app.user])
    monkeypatch.setattr(login_app.module, "user_register", register)
    state = await start(login_app)
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"code": "code", "state": state})
    assert result(response)["auth"] == ["signed-session"]
    assert register.call_args.args[1]["is_superuser"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query,error", [({"error": "access_denied"}, "access_denied"), ({}, "missing_code"), ({"error": "secret-detail"}, "authentication_failed")])
async def test_callback_errors_are_bounded_and_do_not_exchange_tokens(login_app, query, error):
    state = await start(login_app)
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"state": state, **query})
    assert result(response)["error"] == [error]
    login_app.auth.async_exchange_code_for_token.assert_not_awaited()
    login_app.logged_in.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_state_or_channel_cannot_consume_another_login(login_app):
    state = await start(login_app)
    for channel, supplied in [("ngl-auth", "wrong"), ("other", state)]:
        response = await login_app.client.get(f"/api/v1/auth/oauth/{channel}/callback", query_string={"code": "code", "state": supplied})
        assert result(response)["error"] == ["invalid_state"]
    valid = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"code": "code", "state": state})
    assert "auth" in result(valid)


@pytest.mark.asyncio
async def test_expired_transaction_does_not_authenticate(login_app):
    state = await start(login_app)
    async with login_app.client.session_transaction() as state_session:
        state_session["oauth_started_at"] = 1
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"state": state, "code": "code"})
    assert result(response)["error"] == ["invalid_state"]
    login_app.logged_in.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_account_does_not_login(login_app):
    login_app.user.is_active = "0"
    state = await start(login_app)
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"state": state, "code": "code"})
    assert result(response)["error"] == ["user_inactive"]
    login_app.logged_in.assert_not_called()


@pytest.mark.asyncio
async def test_discovery_failure_is_a_retryable_public_result(login_app):
    login_app.factory.side_effect = RuntimeError("private-provider-details")
    response = await login_app.client.get("/api/v1/auth/login/ngl-auth")
    assert result(response)["error"] == ["service_unavailable"]
    assert "private-provider-details" not in response.location


@pytest.mark.asyncio
async def test_exchange_failure_never_exposes_exception(login_app):
    state = await start(login_app)
    login_app.auth.async_exchange_code_for_token.side_effect = RuntimeError("private-token-details")
    response = await login_app.client.get("/api/v1/auth/oauth/ngl-auth/callback", query_string={"state": state, "code": "code"})
    assert result(response)["error"] == ["authentication_failed"]
    assert "private-token-details" not in response.location
