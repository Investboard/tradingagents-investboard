import importlib
import os
import socket
import stat

import pytest
from mcp.shared.auth import OAuthToken

from tradingagents_investboard import auth
from tradingagents_investboard.auth import CALLBACK_PORT, FileTokenStorage, _provider

HOME_VAR = "TRADINGAGENTS_INVESTBOARD_HOME"


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
