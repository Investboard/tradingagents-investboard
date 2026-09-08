"""Turn a completed TradingAgents run into the Investboard ingest payload.

The framework's decision-making agents render structured output back to a
fixed markdown shape (``**Rating**: X`` and friends, see
``tradingagents.agents.schemas``). We read that shape deterministically; an
unrecognisable decision is sent as ``REVIEW`` with the raw text as summary,
never coerced into a tradeable tier.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
TRADER_ACTIONS = ("Buy", "Hold", "Sell")
REPORT_MAX_CHARS = 65_536

_FIELD_RE = re.compile(
    r"\*\*(?P<label>[^*]+)\*\*:\s*(?P<value>.*?)(?=\n\s*\n\*\*|\nFINAL TRANSACTION|\Z)",
    re.DOTALL,
)


def _fields(markdown: str) -> dict[str, str]:
    return {
        m.group("label").strip().lower(): m.group("value").strip()
        for m in _FIELD_RE.finditer(markdown or "")
    }


def _number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _clip(text: str | None) -> str:
    return (text or "")[:REPORT_MAX_CHARS]


def run_id_for(ticker: str, trade_date: str, started_at: datetime) -> str:
    digest = hashlib.sha1(
        f"{ticker}|{trade_date}|{started_at.isoformat()}".encode()
    ).hexdigest()[:8]
    return f"{ticker}_{trade_date}_{digest}"


def parse_decision(markdown: str) -> dict[str, Any]:
    fields = _fields(markdown)
    rating = fields.get("rating", "").strip("* ")
    if rating not in RATINGS:
        return {
            "rating": "REVIEW",
            "executive_summary": _clip(markdown.strip()),
            "investment_thesis": _clip(markdown.strip()),
        }
    decision: dict[str, Any] = {
        "rating": rating,
        "executive_summary": _clip(fields.get("executive summary", "")),
        "investment_thesis": _clip(fields.get("investment thesis", "")),
    }
    target = _number(fields.get("price target"))
    if target is not None and target > 0:
        decision["price_target"] = target
    horizon = fields.get("time horizon")
    if horizon:
        decision["time_horizon"] = horizon[:200]
    return decision


def parse_trader_proposal(markdown: str) -> dict[str, Any] | None:
    if not markdown or not markdown.strip():
        return None
    fields = _fields(markdown)
    action = fields.get("action", "").strip("* ")
    if action not in TRADER_ACTIONS:
        return None
    proposal: dict[str, Any] = {
        "action": action,
        "reasoning": _clip(fields.get("reasoning", "")),
    }
    for key, label in (("entry_price", "entry price"), ("stop_loss", "stop loss")):
        value = _number(fields.get(label))
        if value is not None and value > 0:
            proposal[key] = value
    sizing = fields.get("position sizing")
    if sizing:
        proposal["position_sizing"] = sizing[:200]
    return proposal


def build_run_payload(
    *,
    final_state: dict[str, Any],
    ticker: str,
    trade_date: str,
    config: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
    framework_version: str,
) -> dict[str, Any]:
    debate = final_state.get("investment_debate_state") or {}
    risk = final_state.get("risk_debate_state") or {}
    payload: dict[str, Any] = {
        "framework": "tradingagents",
        "framework_version": framework_version,
        "framework_run_id": run_id_for(ticker, trade_date, started_at),
        "ticker": ticker,
        "as_of_date": trade_date,
        "output_language": str(config.get("output_language") or "English"),
        "model_config": {
            "provider": str(config.get("llm_provider") or "unknown"),
            "deep_think_llm": str(config.get("deep_think_llm") or "unknown"),
            "quick_think_llm": str(config.get("quick_think_llm") or "unknown"),
            "max_debate_rounds": int(config.get("max_debate_rounds") or 0),
            "max_risk_discuss_rounds": int(config.get("max_risk_discuss_rounds") or 0),
        },
        "run_started_at": started_at.isoformat(),
        "run_completed_at": completed_at.isoformat(),
        "decision": parse_decision(final_state.get("final_trade_decision") or ""),
        "reports": {
            "market": _clip(final_state.get("market_report")),
            "sentiment": _clip(final_state.get("sentiment_report")),
            "news": _clip(final_state.get("news_report")),
            "fundamentals": _clip(final_state.get("fundamentals_report")),
        },
        "debate": {
            "bull": _clip(debate.get("bull_history")),
            "bear": _clip(debate.get("bear_history")),
            "research_manager": _clip(debate.get("judge_decision")),
        },
        "risk": {
            "aggressive": _clip(risk.get("aggressive_history")),
            "conservative": _clip(risk.get("conservative_history")),
            "neutral": _clip(risk.get("neutral_history")),
            "portfolio_manager": _clip(risk.get("judge_decision")),
        },
    }
    proposal = parse_trader_proposal(final_state.get("trader_investment_plan") or "")
    if proposal is not None:
        payload["trader_proposal"] = proposal
    return payload
