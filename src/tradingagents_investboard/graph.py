"""TradingAgentsGraph that posts every completed run to Investboard."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .auth import TOKEN_DIR, access_token, base_url
from .client import InvestboardApiError, InvestboardClient
from .payload import build_run_payload

OUTBOX_DIR = TOKEN_DIR / "outbox"

# Refusals, not outages. Retrying them wastes the user's time and, for 429,
# spends the daily cap. 400 belongs here: a malformed payload is malformed on
# every attempt.
TERMINAL_STATUSES = (400, 401, 402, 413, 422, 429)
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
                if error.status in TERMINAL_STATUSES:
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


def write_outbox(payload: dict[str, Any]) -> Path:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    # The outbox and the directory holding it both carry run payloads.
    os.chmod(OUTBOX_DIR.parent, 0o700)
    os.chmod(OUTBOX_DIR, 0o700)
    path = outbox_path_for(str(payload.get("framework_run_id") or ""))
    # Written whole, then renamed into place: a crash mid-write must not leave
    # `replay` a truncated payload to choke on. O_EXCL and 0o600 mean the file
    # is never briefly world-readable and never an existing file we adopt.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
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

    Each entry is judged on its own: a refusal the server will repeat is set
    aside as ``.rejected`` so it cannot block the queue for ever, while a
    transient failure stops the run and leaves every remaining file in place.
    """
    outcome: dict[str, list[str]] = {"sent": [], "rejected": [], "failed": []}
    if not OUTBOX_DIR.exists():
        return outcome
    for path in sorted(OUTBOX_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            path.rename(path.with_name(f"{path.name}.rejected"))
            outcome["rejected"].append(path.stem)
            continue
        run_id = str(payload.get("framework_run_id") or path.stem)
        try:
            post_payload(payload, transport=transport)
        except InvestboardApiError as error:
            if error.status in TERMINAL_STATUSES:
                path.rename(path.with_name(f"{path.name}.rejected"))
                outcome["rejected"].append(run_id)
                continue
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
