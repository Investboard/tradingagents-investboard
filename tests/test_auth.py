import importlib
import json
import os
import socket
import stat
import time
import urllib.parse
import urllib.request

import httpx
import pytest
from mcp.shared.auth import OAuthToken

from tradingagents_investboard import auth
from tradingagents_investboard.auth import (
    CALLBACK_PORT,
    NOT_CONNECTED,
    FileTokenStorage,
    _CallbackServer,
    _provider,
)

HOME_VAR = "TRADINGAGENTS_INVESTBOARD_HOME"
ISSUER = "https://clerk.investboard.de"
TOKEN_ENDPOINT = f"{ISSUER}/oauth/token"
DISCOVERY_PATH = "/.well-known/oauth-authorization-server"


async def test_tokens_are_stored_owner_only(tmp_path):
    storage = FileTokenStorage(tmp_path / "tokens.json")
    assert await storage.get_tokens() is None

    await storage.set_tokens(OAuthToken(access_token="abc", token_type="Bearer"))

    assert stat.S_IMODE(os.stat(storage.path).st_mode) == 0o600
    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "abc"


async def test_the_token_directory_is_owner_only(tmp_path):
    """The tokens file sits in its own directory, which nobody else may read either.

    `TOKEN_DIR` is read from the environment at import time, so the module is
    reloaded around a temporary home and reloaded again afterwards.
    """
    previous = os.environ.get(HOME_VAR)
    os.environ[HOME_VAR] = str(tmp_path / "home")
    reloaded = importlib.reload(auth)
    try:
        storage = reloaded.FileTokenStorage()
        assert storage.path.parent == tmp_path / "home"

        await storage.set_tokens(OAuthToken(access_token="abc", token_type="Bearer"))

        assert stat.S_IMODE(os.stat(storage.path.parent).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(storage.path).st_mode) == 0o600
    finally:
        if previous is None:
            os.environ.pop(HOME_VAR, None)
        else:
            os.environ[HOME_VAR] = previous
        importlib.reload(reloaded)


def test_building_a_provider_does_not_bind_the_callback_port(tmp_path):
    """`replay` asks for a token once per outbox entry, in one process.

    Binding the loopback port when the provider is built made the second
    entry fail with "Address already in use", so the port must stay free
    until the interactive flow actually starts the listener.
    """
    storage = FileTokenStorage(tmp_path / "tokens.json")
    _provider(storage, interactive=False)
    _provider(storage, interactive=False)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", CALLBACK_PORT))
    finally:
        probe.close()


async def test_the_non_interactive_flow_refuses_to_open_a_browser(tmp_path):
    storage = FileTokenStorage(tmp_path / "tokens.json")
    provider = _provider(storage, interactive=False)

    with pytest.raises(RuntimeError, match="tradingagents-investboard connect"):
        await provider.context.redirect_handler("https://app.example/authorize")


def test_the_callback_records_the_code_the_state_and_the_issuer():
    """RFC 9207: Investboard returns `iss`, and the SDK checks it.

    Dropping the parameter here left the client unable to detect an
    authorization-server mix-up, so the redirect is read in full.
    """
    server = _CallbackServer(port=0)
    server.start()
    try:
        query = urllib.parse.urlencode({"code": "code-1", "state": "state-1", "iss": ISSUER})
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/callback?{query}", timeout=5
        ) as response:
            assert response.status == 200
        result = server.wait(timeout=5)
    finally:
        server.stop()

    assert result.code == "code-1"
    assert result.state == "state-1"
    assert result.iss == ISSUER


def _store(tmp_path, monkeypatch, *, obtained_at, access="stored", refresh="refresh-1", **extra):
    monkeypatch.setattr(auth, "TOKEN_DIR", tmp_path)
    monkeypatch.setenv("INVESTBOARD_BASE_URL", "https://app.example")
    storage = auth.FileTokenStorage(tmp_path / "tokens.json")
    data = {
        "tokens": {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": 86400,
            "refresh_token": refresh,
        },
        "tokens_obtained_at": obtained_at,
        "client": {"client_id": "client-1", "redirect_uris": ["http://127.0.0.1:8765/callback"]},
    }
    data.update(extra)
    storage.write(data)
    return storage


def test_a_fresh_access_token_is_returned_without_a_request(tmp_path, monkeypatch):
    """A day-long token is used as it stands; the network is not touched."""
    _store(tmp_path, monkeypatch, obtained_at=time.time() - 60)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    assert auth.access_token(transport=httpx.MockTransport(handler)) == "stored"


def test_an_expired_access_token_is_refreshed_in_a_fresh_process(tmp_path, monkeypatch):
    """The SDK's expiry state is in memory, so a new process must judge for itself.

    `tokens_obtained_at` is what makes that possible: a token stamped a day ago
    is spent, and the refresh token buys a new one without a browser.
    """
    storage = _store(tmp_path, monkeypatch, obtained_at=time.time() - 90_000)
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == DISCOVERY_PATH:
            seen["discovery"] = str(request.url)
            return httpx.Response(200, json={"issuer": ISSUER, "token_endpoint": TOKEN_ENDPOINT})
        seen["token_url"] = str(request.url)
        seen["body"] = dict(urllib.parse.parse_qsl(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed",
                "token_type": "Bearer",
                "expires_in": 86400,
                "refresh_token": "refresh-2",
            },
        )

    assert auth.access_token(transport=httpx.MockTransport(handler)) == "refreshed"

    assert seen["discovery"] == f"https://app.example{DISCOVERY_PATH}"
    assert seen["token_url"] == TOKEN_ENDPOINT
    assert seen["body"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
        "client_id": "client-1",
    }
    stored = json.loads(storage.path.read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "refreshed"
    assert stored["tokens"]["refresh_token"] == "refresh-2"
    assert stored["tokens_obtained_at"] > time.time() - 60
    # Cached, so the next refresh is one request rather than two.
    assert stored["token_endpoint"] == TOKEN_ENDPOINT


def test_a_cached_token_endpoint_skips_discovery(tmp_path, monkeypatch):
    _store(
        tmp_path,
        monkeypatch,
        obtained_at=time.time() - 90_000,
        token_endpoint=TOKEN_ENDPOINT,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != DISCOVERY_PATH
        return httpx.Response(
            200, json={"access_token": "refreshed", "token_type": "Bearer", "expires_in": 86400}
        )

    assert auth.access_token(transport=httpx.MockTransport(handler)) == "refreshed"


def test_a_kept_refresh_token_survives_a_response_that_omits_it(tmp_path, monkeypatch):
    """A server that does not rotate refresh tokens must not cost us ours."""
    storage = _store(
        tmp_path, monkeypatch, obtained_at=time.time() - 90_000, token_endpoint=TOKEN_ENDPOINT
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"access_token": "refreshed", "token_type": "Bearer", "expires_in": 86400}
        )

    auth.access_token(transport=httpx.MockTransport(handler))

    stored = json.loads(storage.path.read_text(encoding="utf-8"))
    assert stored["tokens"]["refresh_token"] == "refresh-1"


def test_a_failed_refresh_reads_as_not_connected(tmp_path, monkeypatch):
    _store(
        tmp_path, monkeypatch, obtained_at=time.time() - 90_000, token_endpoint=TOKEN_ENDPOINT
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    with pytest.raises(RuntimeError, match="tradingagents-investboard connect"):
        auth.access_token(transport=httpx.MockTransport(handler))


def test_no_stored_token_reads_as_not_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "TOKEN_DIR", tmp_path / "empty")

    with pytest.raises(RuntimeError) as excinfo:
        auth.access_token()

    assert str(excinfo.value) == NOT_CONNECTED
