"""Turn a completed TradingAgents run into the Investboard ingest payload.

The framework's decision-making agents render structured output back to a
fixed markdown shape (``**Rating**: X`` and friends, see
``tradingagents.agents.schemas``). We read that shape deterministically; a
decision whose labels we cannot read keeps its raw text as the summary, whether
the rating comes from the framework's own extractor or is ``REVIEW``. Nothing
unreadable is coerced into a tradeable tier.

Two parsing rules earn their keep against real model output. A field ends only
where another *known* label begins, so a bold heading the model invents inside
a thesis does not truncate it; and the first occurrence of a label wins, so a
rating quoted inside a thesis cannot overwrite the real one.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
TRADER_ACTIONS = ("Buy", "Hold", "Sell")
REPORT_MAX_CHARS = 65_536

# Every label the rendered decision and trader plan can carry. The set is both
# what we read and what ends the value before it.
KNOWN_LABELS = (
    "Rating",
    "Executive Summary",
    "Investment Thesis",
    "Price Target",
    "Time Horizon",
    "Action",
    "Reasoning",
    "Entry Price",
    "Stop Loss",
    "Position Sizing",
)

_LABELS = "|".join(re.escape(label) for label in KNOWN_LABELS)
_TERMINATOR = rf"(?=\n[ \t]*\*\*[ \t]*(?:{_LABELS})[ \t]*\*\*[ \t]*:|\n\s*FINAL TRANSACTION|\Z)"
_FIELD_RE = re.compile(
    rf"\*\*[ \t]*(?P<label>{_LABELS})[ \t]*\*\*[ \t]*:[ \t]*(?P<value>.*?){_TERMINATOR}",
    re.DOTALL | re.IGNORECASE,
)

# A leading currency marker: one symbol, or an ISO-style three-letter code.
_CURRENCY_PREFIX_RE = re.compile(r"^(?:[$€£¥₣₹₺₩]|[A-Za-z]{3}\b)[ \t]*")
# What is left has to be one number and nothing else: no second figure, no
# range, no trailing prose.
_BARE_NUMBER_RE = re.compile(r"\d[\d.,]*")


def _fields(markdown: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _FIELD_RE.finditer(markdown or ""):
        # setdefault, not assignment: the first occurrence of a label is the
        # agent's own; a later one is quoted inside somebody's argument.
        fields.setdefault(match.group("label").strip().lower(), match.group("value").strip())
    return fields


def _is_grouped(digits: str, separator: str) -> bool:
    """True when ``digits`` reads as thousands grouping: 1, 1.234, 12.345.678."""
    parts = digits.split(separator)
    if len(parts) < 2 or not (1 <= len(parts[0]) <= 3) or not parts[0].isdigit():
        return False
    return all(len(part) == 3 and part.isdigit() for part in parts[1:])


def _normalise_separators(text: str) -> str | None:
    """Rewrite a grouped/decimal number as a plain float literal, or refuse.

    A lone dot is a decimal point. The framework renders a price target with
    ``str(float)``, so that is the only reading it can have; treating it as a
    thousands group made every sub-ten price a thousandfold error.

    A lone comma carries no such guarantee, and there refusing beats guessing:
    "1,25" is either 1.25 or a broken group, and a hundredfold error in a price
    target is worse than no price target at all.
    """
    has_dot = "." in text
    has_comma = "," in text
    if not has_dot and not has_comma:
        return text
    if has_dot and has_comma:
        # Both present: the last one is the decimal mark, the other must group.
        decimal = "." if text.rfind(".") > text.rfind(",") else ","
        group = "," if decimal == "." else "."
        head, _, tail = text.rpartition(decimal)
        if decimal in head or not tail.isdigit() or not _is_grouped(head, group):
            return None
        return f"{head.replace(group, '')}.{tail}"
    separator = "." if has_dot else ","
    if text.count(separator) > 1:
        # Several of the same separator can only be grouping.
        return text.replace(separator, "") if _is_grouped(text, separator) else None
    if separator == ".":
        return text if text.partition(".")[2].isdigit() else None
    # One comma: a thousands group is the only unambiguous reading it has, and
    # that is 1 to 3 digits ahead of exactly 3 behind, nothing else.
    return text.replace(",", "") if _is_grouped(text, ",") else None


def _number(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    text = _CURRENCY_PREFIX_RE.sub("", text, count=1).strip()
    sign = -1.0 if text.startswith("-") else 1.0
    if text[:1] in {"+", "-"}:
        text = text[1:].strip()
    if not _BARE_NUMBER_RE.fullmatch(text):
        return None
    normalised = _normalise_separators(text)
    if normalised is None:
        return None
    try:
        return sign * float(normalised)
    except ValueError:
        return None


def _clip(text: str | None) -> str:
    return (text or "")[:REPORT_MAX_CHARS]


def _utc(moment: datetime) -> str:
    """UTC with a Z suffix: the wire form the ingest contract asks for."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _heuristic_rating(markdown: str) -> str | None:
    """The framework's own rating extractor, used only when the labels fail.

    Imported inside the function so this module stays importable (and testable)
    without the framework installed.
    """
    try:
        from tradingagents.agents.utils.rating import extract_rating
    except Exception:  # the framework is optional at import time
        return None
    try:
        return extract_rating(markdown)
    except Exception:
        return None


def run_id_for(ticker: str, trade_date: str, started_at: datetime) -> str:
    digest = hashlib.sha1(f"{ticker}|{trade_date}|{started_at.isoformat()}".encode()).hexdigest()[
        :8
    ]
    return f"{ticker}_{trade_date}_{digest}"


def parse_decision(markdown: str) -> dict[str, Any]:
    fields = _fields(markdown)
    rating = fields.get("rating", "").strip("* ").strip()
    if rating not in RATINGS:
        # The labels failed. The framework's own extractor may still read a
        # rating out of the prose, but that prose is then the only reasoning
        # there is, and a tradeable rating must never travel without it. Both
        # endings therefore keep the raw text once, as the summary: repeating
        # it as a thesis would dress a parse failure up as an argument.
        heuristic = _heuristic_rating(markdown or "")
        return {
            "rating": heuristic if heuristic in RATINGS else "REVIEW",
            "executive_summary": _clip((markdown or "").strip()),
            "investment_thesis": "",
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
        "run_started_at": _utc(started_at),
        "run_completed_at": _utc(completed_at),
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
