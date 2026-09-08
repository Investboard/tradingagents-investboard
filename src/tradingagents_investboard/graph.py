"""TradingAgentsGraph that posts every completed run to Investboard."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.graph.trading_graph import TradingAgentsGraph

from .auth import TOKEN_DIR, access_token, base_url
from .client import InvestboardApiError, InvestboardClient
from .payload import build_run_payload

OUTBOX_DIR = TOKEN_DIR / "outbox"


def framework_version() -> str:
    try:
        from importlib.metadata import version

        return version("tradingagents")
    except Exception:  # noqa: BLE001 (best-effort label)
        return "unknown"


def post_payload(payload: dict[str, Any]) -> dict[str, Any]:
    client = InvestboardClient(base_url(), access_token())
    try:
        for attempt in range(3):
            try:
                return client.post_run(payload)
            except InvestboardApiError as error:
                if error.status in (401, 402, 413, 422, 429):
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


def write_outbox(payload: dict[str, Any]) -> Path:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTBOX_DIR / f"{payload['framework_run_id']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def replay_outbox() -> list[str]:
    sent: list[str] = []
    for path in sorted(OUTBOX_DIR.glob("*.json")) if OUTBOX_DIR.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        post_payload(payload)
        path.unlink()
        sent.append(payload["framework_run_id"])
    return sent


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
        except Exception as error:  # noqa: BLE001 (the run itself succeeded; `replay` retries the post)
            path = write_outbox(payload)
            final_state["investboard"] = {"stored": None, "outbox": str(path), "error": str(error)}
        return final_state, signal
