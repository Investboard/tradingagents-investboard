"""Clerk OAuth for Investboard, through the MCP Python SDK's client flow.

Investboard's MCP server publishes its authorization-server metadata; the SDK
discovers it, registers this CLI as a public PKCE client, opens the browser
for consent and stores the tokens here. Access tokens live one day and the
refresh token does not expire, so every run re-opens a short MCP session
first: the SDK refreshes when needed and the stored access token is then
handed to the REST client.

The HTTP client for that session comes from ``create_mcp_http_client``: from
mcp 2.x the transports speak httpx2, and ``OAuthClientProvider`` is an
``httpx2.Auth``, so a plain ``httpx.AsyncClient`` cannot carry it. The REST
client in ``client.py`` stays on httpx.
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

from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

DEFAULT_BASE_URL = "https://app.investboard.de"
CALLBACK_PORT = 8765
TOKEN_DIR = Path(
    os.environ.get(
        "TRADINGAGENTS_INVESTBOARD_HOME", Path.home() / ".tradingagents" / "investboard"
    )
)


def base_url() -> str:
    return os.environ.get("INVESTBOARD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


class FileTokenStorage(TokenStorage):
    def __init__(self, path: Path | None = None):
        self.path = path or (TOKEN_DIR / "tokens.json")

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(self.path, 0o600)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(exclude_none=True)
        data["tokens_obtained_at"] = time.time()
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client"] = client_info.model_dump(exclude_none=True, mode="json")
        self._write(data)


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
        self.result: dict[str, str | None] = {"code": None, "state": None, "error": None}
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # http.server names its handlers this way
                query = parse_qs(urlparse(self.path).query)
                if "code" in query:
                    outer.result["code"] = query["code"][0]
                    outer.result["state"] = query.get("state", [None])[0]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><p>Investboard is connected. "
                        b"You can return to the terminal.</p></body></html>"
                    )
                else:
                    outer.result["error"] = query.get("error", ["unknown"])[0]
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, *args):  # silence
                return

        self._handler = Handler

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = HTTPServer(("127.0.0.1", self.port), self._handler)
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
                    code=self.result["code"], state=self.result["state"], iss=None
                )
            if self.result["error"]:
                raise RuntimeError(f"Authorization failed: {self.result['error']}")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for the browser to return")


def _provider(storage: FileTokenStorage, interactive: bool) -> OAuthClientProvider:
    server = _CallbackServer()

    async def redirect(url: str) -> None:
        if not interactive:
            raise RuntimeError("Not connected. Run: tradingagents-investboard connect")
        server.start()
        print("Opening your browser to connect Investboard")
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
    """Interactive first-time connection."""
    asyncio.run(_touch_session(interactive=True))


def access_token() -> str:
    """A fresh access token for the REST client; refreshes through the MCP session, never prompts."""
    return asyncio.run(_touch_session(interactive=False)).access_token
