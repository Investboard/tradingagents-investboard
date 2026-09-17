"""Clerk OAuth for Investboard, through the MCP Python SDK's client flow.

Investboard's MCP server publishes its authorization-server metadata; the SDK
discovers it, registers this CLI as a public PKCE client, opens the browser for
consent and stores the tokens here. That happens once, in ``connect``.

Later runs never open an MCP session. Access tokens live one day and the
refresh token is long-lived, so ``access_token`` reads the stored pair,
returns the access token while it is still fresh, and otherwise exchanges the
refresh token at the authorization server's token endpoint itself. The SDK
keeps its expiry bookkeeping in memory, which is no help to a fresh process;
the ``tokens_obtained_at`` stamp written beside the tokens is.

The HTTP client for the ``connect`` session comes from
``create_mcp_http_client``: from mcp 2.x the transports speak httpx2, and
``OAuthClientProvider`` is an ``httpx2.Auth``, so a plain ``httpx.AsyncClient``
cannot carry it. The REST client in ``client.py``, and the refresh below, stay
on httpx.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

DEFAULT_BASE_URL = "https://app.investboard.de"
CALLBACK_PORT = 8765
TOKEN_DIR = Path(
    os.environ.get("TRADINGAGENTS_INVESTBOARD_HOME", Path.home() / ".tradingagents" / "investboard")
)
NOT_CONNECTED = "Not connected. Run: tradingagents-investboard connect"
# A refusal of the connection and a failure to ask are different problems with
# different remedies, so they are not reported as one. Connecting again cannot
# fix an outage, and it costs a browser round trip to find that out.
UNREACHABLE = (
    "Investboard could not be reached while refreshing the connection. Try again in a moment."
)
# Refresh a little before the server would reject the token, so a long run does
# not start with one that expires mid-flight.
REFRESH_LEEWAY_SECONDS = 60


def base_url() -> str:
    return os.environ.get("INVESTBOARD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


class FileTokenStorage(TokenStorage):
    def __init__(self, path: Path | None = None):
        self.path = path or (TOKEN_DIR / "tokens.json")

    def read(self) -> dict:
        """The whole token file: tokens, client registration, cached endpoint."""
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # An explicit chmod, not mkdir(mode=...): that mode argument is masked
        # by the umask, and ignored outright when the directory already exists.
        os.chmod(self.path.parent, 0o700)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(self.path, 0o600)

    def store_tokens(self, tokens: OAuthToken) -> None:
        """Persist a token pair and stamp when it was obtained.

        The stamp is what makes expiry legible to a later process: the SDK's own
        expiry state lives in memory and dies with the session that created it.
        """
        data = self.read()
        data["tokens"] = tokens.model_dump(exclude_none=True)
        data["tokens_obtained_at"] = time.time()
        self.write(data)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self.read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.store_tokens(tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self.read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self.read()
        data["client"] = client_info.model_dump(exclude_none=True, mode="json")
        self.write(data)


def parse_callback_query(raw_query: str) -> dict[str, str | None]:
    """Read the authorization response out of the redirect's query string.

    ``iss`` is RFC 9207. Investboard's metadata advertises
    ``authorization_response_iss_parameter_supported``, and the SDK compares the
    value against the discovered issuer, so dropping it here left the client
    unable to notice a mix-up. It is carried through.
    """
    query = parse_qs(raw_query)
    return {
        "code": query.get("code", [None])[0],
        "state": query.get("state", [None])[0],
        "iss": query.get("iss", [None])[0],
        "error": None if "code" in query else query.get("error", ["unknown"])[0],
    }


class _CallbackServer:
    """Loopback listener for the OAuth redirect.

    The port is bound in :meth:`start`, not here. Every command builds a
    provider, but only the interactive ``connect`` flow ever needs the
    listener; binding eagerly made a second provider in the same process
    (the ``replay`` loop, which asks for a token once per outbox entry)
    fail with "Address already in use".
    """

    def __init__(self, port: int = CALLBACK_PORT):
        self.port = port
        self.result: dict[str, str | None] = {
            "code": None,
            "state": None,
            "iss": None,
            "error": None,
        }
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # http.server names its handlers this way
                parsed = parse_callback_query(urlparse(self.path).query)
                if parsed["code"]:
                    outer.result.update(parsed)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><p>Investboard is connected. "
                        b"You can return to the terminal.</p></body></html>"
                    )
                else:
                    outer.result["error"] = parsed["error"] or "unknown"
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, *args):  # silence
                return

        self._handler = Handler

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = HTTPServer(("127.0.0.1", self.port), self._handler)
        # The bound port, which is `CALLBACK_PORT` on every real connection: it
        # has to match the redirect URI the client registered. Only the tests
        # pass port 0 and let the OS pick, so they read back what it picked.
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def wait(self, timeout: float = 300.0) -> AuthorizationCodeResult:
        if self._server is None:
            raise RuntimeError("The callback listener was never started")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.result["code"]:
                return AuthorizationCodeResult(
                    code=self.result["code"],
                    state=self.result["state"],
                    iss=self.result["iss"],
                )
            if self.result["error"]:
                raise RuntimeError(f"Authorization failed: {self.result['error']}")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for the browser to return")


def _provider(storage: FileTokenStorage, interactive: bool) -> OAuthClientProvider:
    server = _CallbackServer()

    async def redirect(url: str) -> None:
        if not interactive:
            raise RuntimeError(NOT_CONNECTED)
        server.start()
        # Printed before the browser is opened: on a headless machine the open
        # silently does nothing, and the URL is all the user has to work with.
        print("Open this URL to connect Investboard:")
        print(url)
        webbrowser.open(url)

    async def callback() -> AuthorizationCodeResult:
        try:
            return server.wait()
        finally:
            server.stop()

    return OAuthClientProvider(
        server_url=base_url(),
        client_metadata=OAuthClientMetadata.model_validate(
            {
                "client_name": "TradingAgents connect",
                "redirect_uris": [f"http://127.0.0.1:{CALLBACK_PORT}/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            }
        ),
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )


async def _touch_session(interactive: bool) -> OAuthToken:
    storage = FileTokenStorage()
    provider = _provider(storage, interactive)
    async with (
        create_mcp_http_client(auth=provider) as http,
        streamable_http_client(url=f"{base_url()}/mcp", http_client=http) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
    tokens = await storage.get_tokens()
    if tokens is None:
        raise RuntimeError("Connection did not produce a token")
    return tokens


def connect() -> None:
    """Interactive first-time connection: browser consent, registration, tokens."""
    asyncio.run(_touch_session(interactive=True))


def _discover_token_endpoint(http: httpx.Client) -> str:
    response = http.get(f"{base_url()}/.well-known/oauth-authorization-server")
    response.raise_for_status()
    endpoint = response.json().get("token_endpoint")
    if not endpoint:
        raise RuntimeError("The authorization-server metadata carries no token_endpoint")
    return str(endpoint)


def _is_invalid_grant(response: httpx.Response) -> bool:
    """True when the server refused the refresh token itself.

    That is the one failure ``connect`` fixes. Everything else the endpoint can
    answer with is about the request or the server, and re-consenting does not
    touch it.
    """
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error") == "invalid_grant"


def _refresh(storage: FileTokenStorage, data: dict, transport: httpx.BaseTransport | None) -> str:
    tokens = data.get("tokens") or {}
    refresh_token = tokens.get("refresh_token")
    client = data.get("client") or {}
    client_id = client.get("client_id")
    if not refresh_token or not client_id:
        raise RuntimeError(NOT_CONNECTED)
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    # How the client authenticates was settled at registration. This one asks
    # to be a public PKCE client ("none"), where the client id is the whole
    # identity, but a server may register it as `client_secret_post` instead
    # and then refuse every exchange that arrives without the secret it issued.
    if client.get("token_endpoint_auth_method") == "client_secret_post" and client.get(
        "client_secret"
    ):
        body["client_secret"] = str(client["client_secret"])
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0), transport=transport) as http:
            endpoint = data.get("token_endpoint") or _discover_token_endpoint(http)
            response = http.post(endpoint, data=body, headers={"accept": "application/json"})
            response.raise_for_status()
            fresh = OAuthToken.model_validate(response.json())
    except httpx.HTTPStatusError as error:
        if _is_invalid_grant(error.response):
            raise RuntimeError(NOT_CONNECTED) from error
        raise RuntimeError(UNREACHABLE) from error
    except Exception as error:
        raise RuntimeError(UNREACHABLE) from error
    # A server that does not rotate the refresh token omits it from the
    # response; keeping the stored one is what lets the next run refresh again.
    if fresh.refresh_token is None:
        fresh.refresh_token = refresh_token
    data["tokens"] = fresh.model_dump(exclude_none=True)
    data["tokens_obtained_at"] = time.time()
    data["token_endpoint"] = endpoint
    storage.write(data)
    return fresh.access_token


def access_token(transport: httpx.BaseTransport | None = None) -> str:
    """A usable access token, without a browser and without an MCP session.

    Reads what ``connect`` stored, hands back the access token while it is still
    fresh, and otherwise spends the refresh token. A missing or refused
    connection is reported as such, because ``connect`` is the remedy; a
    refresh that never got an answer is reported as an outage, because it is
    not.
    """
    storage = FileTokenStorage()
    data = storage.read()
    tokens = data.get("tokens") or {}
    stored = tokens.get("access_token")
    if not stored:
        raise RuntimeError(NOT_CONNECTED)
    obtained_at = data.get("tokens_obtained_at")
    expires_in = tokens.get("expires_in")
    if obtained_at and expires_in:
        expiry = float(obtained_at) + float(expires_in) - REFRESH_LEEWAY_SECONDS
        if time.time() < expiry:
            return str(stored)
    return _refresh(storage, data, transport)
