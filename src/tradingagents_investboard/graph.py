"""TradingAgentsGraph that posts every completed run to Investboard."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tradingagents.graph.trading_graph import TradingAgentsGraph

from . import context
from .auth import TOKEN_DIR, access_token, base_url
from .client import InvestboardApiError, InvestboardClient
from .payload import build_run_payload

OUTBOX_DIR = TOKEN_DIR / "outbox"

# Refusals, not outages. Retrying one wastes the user's time and, for 429,
# spends the daily cap. They part company only in the outbox.
#
# A permanent refusal is about the payload, and the server will repeat it
# however long we wait: it cannot parse the payload, cannot resolve the
# subject, or the payload is over the size limit.
PERMANENT_STATUSES = (400, 413, 422)
# A requeue refusal is about the account, not the payload: a connection that
# has lapsed, a paused subscription, a daily cap already spent. The payload is
# good and will be accepted once the account is, so it must keep its place in
# the queue rather than be set aside for the user to rename by hand.
REQUEUE_STATUSES = (401, 402, 429)
# Either way, a single post does not retry itself.
NO_RETRY_STATUSES = PERMANENT_STATUSES + REQUEUE_STATUSES
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def framework_version() -> str:
    try:
        from importlib.metadata import version

        return version("tradingagents")
    except Exception:  # best-effort label
        return "unknown"


def post_payload(
    payload: dict[str, Any], transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    client = InvestboardClient(base_url(), access_token(), transport=transport)
    try:
        for attempt in range(3):
            try:
                return client.post_run(payload)
            except InvestboardApiError as error:
                if error.status in NO_RETRY_STATUSES:
                    raise
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
            except Exception:  # network failure, retried below
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")
    finally:
        client.close()


def outbox_path_for(run_id: str) -> Path:
    """Where one run parks. The run id reaches us from model output, so it is
    sanitised rather than trusted as a path component."""
    safe = _UNSAFE_IN_FILENAME.sub("_", run_id) or "run"
    return OUTBOX_DIR / f"{safe}.json"


def _set_aside(path: Path) -> Path:
    """Rename an entry out of the queue, without overwriting an earlier one.

    Two runs of the same instrument on the same date can be refused twice, and
    the second refusal must not silently delete the first entry: nothing in the
    outbox is ever thrown away.
    """
    target = path.with_name(f"{path.name}.rejected")
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.name}.{stamp}.rejected")
    path.rename(target)
    return target


def write_outbox(payload: dict[str, Any]) -> Path:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    # The outbox and the directory holding it both carry run payloads.
    os.chmod(OUTBOX_DIR.parent, 0o700)
    os.chmod(OUTBOX_DIR, 0o700)
    path = outbox_path_for(str(payload.get("framework_run_id") or ""))
    # Written whole, then renamed into place: a crash mid-write must not leave
    # `replay` a truncated payload to choke on. O_EXCL and 0o600 mean the file
    # is never briefly world-readable and never an existing file we adopt.
    # A random suffix, not the pid: two writers can share a pid across a fork
    # or a container restart, and O_EXCL would then fail on a stale temporary.
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def replay_outbox(transport: httpx.BaseTransport | None = None) -> dict[str, list[str]]:
    """Post what the outbox holds, one entry at a time.

    Each entry is judged on its own. A refusal about the payload is permanent,
    so the entry is set aside as ``.rejected`` and cannot block the queue for
    ever. A refusal about the account, and any transient failure, stops the run
    and leaves every remaining file in place: an expired connection, a paused
    subscription and a spent daily cap all clear on their own, and asking the
    user to rename a file afterwards is how a good run gets lost.
    """
    outcome: dict[str, list[str]] = {"sent": [], "rejected": [], "failed": []}
    if not OUTBOX_DIR.exists():
        return outcome
    for path in sorted(OUTBOX_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            _set_aside(path)
            outcome["rejected"].append(path.stem)
            continue
        run_id = str(payload.get("framework_run_id") or path.stem)
        try:
            post_payload(payload, transport=transport)
        except InvestboardApiError as error:
            if error.status in PERMANENT_STATUSES:
                _set_aside(path)
                outcome["rejected"].append(run_id)
                continue
            # A requeue status, or a status we have no policy for: keep it.
            outcome["failed"].append(run_id)
            break
        except Exception:
            outcome["failed"].append(run_id)
            break
        path.unlink()
        outcome["sent"].append(run_id)
    return outcome


class InvestboardTradingAgentsGraph(TradingAgentsGraph):
    """Same graph, plus one post to Investboard after each completed run."""

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        """The framework's paragraph, then the owner's policy and position.

        The framework calls this once at the start of a run and threads the
        result to every agent, so one client is built here and closed again
        rather than held: these two reads are the only ones it serves.

        The base method is fail-open by design, and this override deliberately
        is not. Its identity lookup returns ``{}`` when yfinance cannot answer,
        so the run continues on ticker-only context rather than failing before
        analysis starts. A missing mandate is not the same kind of gap: a
        mandate-blind run is not a degraded run, it is a run whose report the
        user will read as mandate-checked. So a policy or position read that
        cannot be served stops the run here, rather than dropping the owner's
        mandate out of every agent's context without saying so.
        """
        base = super().resolve_instrument_context(ticker, asset_type)
        client = InvestboardClient(base_url(), access_token())
        try:
            return base + context.instrument_context_blocks(client, ticker)
        finally:
            client.close()

    def propagate(self, company_name, trade_date, asset_type: str = "stock"):
        started_at = datetime.now(timezone.utc)
        final_state, signal = super().propagate(company_name, trade_date, asset_type=asset_type)
        completed_at = datetime.now(timezone.utc)
        payload = build_run_payload(
            final_state=final_state,
            ticker=company_name,
            trade_date=str(trade_date),
            config=self.config,
            started_at=started_at,
            completed_at=completed_at,
            framework_version=framework_version(),
        )
        try:
            stored = post_payload(payload)
            final_state["investboard"] = {"stored": stored, "outbox": None}
        except Exception as error:  # the run itself succeeded; `replay` retries the post
            path = write_outbox(payload)
            final_state["investboard"] = {"stored": None, "outbox": str(path), "error": str(error)}
        return final_state, signal
