"""The one seam every Investboard read goes through: client, token, errors.

The vendor methods above this module are rendering and nothing else. The HTTP
client they call, the freshness of the token it carries and the translation of
a refusal into the framework's error taxonomy live here, so there is one place
to reason about the connection and one place for a test to route.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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

    That safety is bought, and the price belongs beside it. A refresh runs
    inside the lock, and `auth` gives it a 30s timeout on every phase, so up to
    two 30s exchanges (the connect, then the read of the token endpoint) can be
    held across it. For that whole time every analyst thread waits here, not
    only the ones that would have refreshed: a slow authorization server stalls
    the run rather than one call in it. Outside a refresh the standing cost is
    a token-file read and a JSON parse per vendor call, serialised through the
    same lock. Both are accepted deliberately, because the alternatives are a
    dead bearer mid-analysis and a refresh a thread.

    A superseded client is not closed on the spot: a thread that took it before
    the rotation may still be mid-request on it. It is held for one generation
    and closed at the next rotation. What that guarantees is one rotation of
    grace, which is not the same as safety: two rotations inside a single
    in-flight request still close a client under its caller. An analysis long
    enough to cross two token refreshes, or a `_reset_client` beside a running
    read, does exactly that, and httpx answers the next send on that client
    with `RuntimeError: Cannot send a request, as the client has been closed`.
    `_translate` reads that one as a "send it again" rather than letting it
    surface as a crash with nothing in it for the reader to act on.
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


CLIENT_CLOSED = "client has been closed"


def _named(error: InvestboardApiError) -> str:
    """The refusal with its reason named exactly once.

    The reason is the word the CLI turns into a next step, so every branch that
    reports a refusal has to carry it: without it `daily_cap_reached` and
    `provider_unavailable` are the same sentence to the user. But `_unwrap`
    already folds `details` into the message where the server sent them, and
    `details.reason` is normally that same word, so a fixed prefix prints it
    beside itself. The prefix is added only where the message lacks it.
    """
    if not error.reason or error.reason in error.message:
        return error.message
    return f"{error.reason}: {error.message}"


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
        if CLIENT_CLOSED in str(error):
            # What httpx raises when a rotation closed the client under an
            # in-flight read (see `_client`). Nothing about the read itself
            # failed and the next attempt takes the current client, so this is
            # a "send it again", not a configuration the user has to fix.
            return VendorRateLimitError(
                f"Investboard: the client was rotated mid-request, send the read again: {error}"
            )
        return error
    if isinstance(error, InvestboardApiError):
        if error.status in (429, 503):
            # `_unwrap` folds `details` into the message on a 4xx only, so a
            # 503 arrives without its reason: without naming it here, a daily
            # cap and a provider outage read as the same sentence.
            return VendorRateLimitError(f"Investboard: {_named(error)}")
        if error.status in (401, 402, 403):
            return VendorNotConfiguredError(f"Investboard: {_named(error)}")
        if error.status == 422:
            return NoMarketDataError(symbol, None, error.message)
    return error


def _call(symbol: str, fn: Callable[[InvestboardClient], Any]) -> Any:
    """One read, with a refusal translated on its way out and a rate limit logged.

    The log line is what a rate limit leaves behind. ``route_to_vendor`` keeps a
    ``VendorNotConfiguredError`` as its ``first_error`` and raises it when the
    chain is exhausted, so that message reaches the user; it keeps nothing for a
    ``VendorRateLimitError``, which only moves it to the next vendor. The CLI
    names Investboard as the only vendor for prices, indicators and
    fundamentals, so there is no next vendor there and the framework ends the
    chain with a bare ``RuntimeError("No available vendor for ...")``. The daily
    cap, the provider outage, the reason and any wait the server named all
    disappear into that sentence, so the translated message is written to the
    run's log before the raise, where the reader can still act on it.
    """
    try:
        return fn(_client())
    except (InvestboardApiError, RuntimeError, httpx.TransportError) as error:
        translated = _translate(error, symbol)
        if translated is error:
            raise
        if isinstance(translated, VendorRateLimitError):
            logger.warning("%s", translated)
        raise translated from error
