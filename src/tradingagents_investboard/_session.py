"""The one seam every Investboard read goes through: client, token, errors.

The vendor methods above this module are rendering and nothing else. The HTTP
client they call, the freshness of the token it carries and the translation of
a refusal into the framework's error taxonomy live here, so there is one place
to reason about the connection and one place for a test to route.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import httpx
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)

from .auth import NOT_CONNECTED, UNREACHABLE, access_token, base_url
from .client import InvestboardApiError, InvestboardClient

# Test seam: a MockTransport routed through here never reads the token file.
_transport: httpx.BaseTransport | None = None
_client_instance: InvestboardClient | None = None
# The client the last rotation replaced, kept one generation before it is
# closed; see `_client`.
_client_retired: InvestboardClient | None = None
_client_token: str | None = None
_client_lock = threading.Lock()


def _reset_client() -> None:
    """Drop the singleton and close both pools; no request is in flight here."""
    global _client_instance, _client_retired, _client_token
    with _client_lock:
        closing = (_client_instance, _client_retired)
        _client_instance = None
        _client_retired = None
        _client_token = None
    for client in closing:
        if client is not None:
            client.close()


def _client() -> InvestboardClient:
    """One client per process, rebuilt when the access token has moved on.

    The client is worth keeping: it holds the connection pool. The token it was
    built with is not, because an access token lives a day and an analysis can
    outlive it, so the token is read on every call and kept beside the
    instance; a different one rebuilds the client.

    The read happens under the same lock as the rebuild, and that is the whole
    of the safety. Read outside it, a thread can take a token, be descheduled
    while another refreshes and rebuilds, then take the lock and rebuild the
    singleton with the expired one it is still holding: the next request goes
    out with a dead bearer, the server answers 401 and the framework quietly
    serves that call from another vendor. Holding the lock across the read also
    means one refresh exchange rather than one a thread, which matters where
    the refresh token rotates and every loser gets `invalid_grant`.

    A superseded client is not closed on the spot: a thread that took it before
    the rotation may still be mid-request on it. It is held for one generation
    and closed at the next rotation, by which time anything it was serving has
    finished or hit the 60s read timeout many times over.
    """
    global _client_instance, _client_retired, _client_token
    with _client_lock:
        token = access_token()
        superseded: InvestboardClient | None = None
        if _client_instance is None or _client_token != token:
            superseded = _client_retired
            _client_retired = _client_instance
            _client_instance = InvestboardClient(base_url(), token, transport=_transport)
            _client_token = token
        instance = _client_instance
    if superseded is not None:
        superseded.close()
    return instance


def _translate(error: Exception, symbol: str) -> Exception:
    """The framework's own error taxonomy, so its vendor chain can act on ours.

    A cap, a provider outage or a transport that never answered is a rate limit
    to the router (next vendor, or wait). A credential fault, a paused account,
    an instrument outside the data scope and a connection that was never made
    are a configuration the user has to fix, and the message carries the
    server's hint so the CLI prints the next step. An instrument the provider
    does not carry is no market data.
    """
    if isinstance(error, httpx.TransportError):
        return VendorRateLimitError(f"Investboard: the read did not complete: {error}")
    if isinstance(error, RuntimeError):
        # What `auth` raises: a connection that was never made or was revoked,
        # where `connect` is the remedy, and a refresh that got no answer,
        # where it is not and waiting is.
        if str(error) == NOT_CONNECTED:
            return VendorNotConfiguredError(f"Investboard: {error}")
        if str(error) == UNREACHABLE:
            return VendorRateLimitError(f"Investboard: {error}")
        return error
    if isinstance(error, InvestboardApiError):
        if error.status in (429, 503):
            return VendorRateLimitError(f"Investboard: {error.message}")
        if error.status in (401, 402, 403):
            # The reason is the word the CLI turns into a next step, so it is
            # named here rather than left to `_unwrap`, which only folds it
            # into the message when the server sent `details`.
            return VendorNotConfiguredError(f"Investboard: {error.reason}: {error.message}")
        if error.status == 422:
            return NoMarketDataError(symbol, None, error.message)
    return error


def _call(symbol: str, fn: Callable[[InvestboardClient], Any]) -> Any:
    try:
        return fn(_client())
    except (InvestboardApiError, RuntimeError, httpx.TransportError) as error:
        translated = _translate(error, symbol)
        if translated is error:
            raise
        raise translated from error
