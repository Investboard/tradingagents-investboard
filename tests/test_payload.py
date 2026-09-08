from datetime import datetime, timedelta, timezone

import pytest

from tradingagents_investboard.payload import (
    _number,
    build_run_payload,
    parse_decision,
    run_id_for,
)

STARTED = datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc)
ENDED = datetime(2026, 9, 8, 9, 10, tzinfo=timezone.utc)
BERLIN = timezone(timedelta(hours=2))


def build(final_state, run_config, **overrides):
    kwargs = {
        "final_state": final_state,
        "ticker": "SAP.DE",
        "trade_date": "2026-09-08",
        "config": run_config,
        "started_at": STARTED,
        "completed_at": ENDED,
        "framework_version": "0.4.1",
    }
    kwargs.update(overrides)
    return build_run_payload(**kwargs)


def test_run_id_is_deterministic_per_ticker_date_and_start():
    a = run_id_for("SAP.DE", "2026-09-08", STARTED)
    b = run_id_for("SAP.DE", "2026-09-08", STARTED)
    c = run_id_for("SAP.DE", "2026-09-08", ENDED)
    assert a == b
    assert a != c
    assert a.startswith("SAP.DE_2026-09-08_")


def test_payload_reads_the_rendered_decision_and_proposal(final_state, run_config):
    payload = build(final_state, run_config)
    assert payload["framework"] == "tradingagents"
    assert payload["framework_version"] == "0.4.1"
    assert payload["ticker"] == "SAP.DE"
    assert payload["as_of_date"] == "2026-09-08"
    assert payload["decision"] == {
        "rating": "Overweight",
        "executive_summary": "Constructive view over the next quarter.",
        "investment_thesis": "Cloud revenue growth is durable.",
        "price_target": 260.0,
        "time_horizon": "3-6 months",
    }
    assert payload["trader_proposal"] == {
        "action": "Buy",
        "reasoning": "Momentum and fundamentals agree.",
        "entry_price": 241.5,
        "stop_loss": 225.0,
        "position_sizing": "3% of portfolio",
    }
    assert payload["debate"]["bull"] == "Bull Analyst: up"
    assert payload["risk"]["portfolio_manager"].startswith("**Rating**")
    assert payload["model_config"]["deep_think_llm"] == "claude-opus-5"
    assert payload["run_started_at"] == "2026-09-08T09:00:00Z"


@pytest.mark.parametrize(
    "started,completed",
    [
        # Naive, as a caller who forgot the timezone would pass it.
        (datetime(2026, 9, 8, 9, 0), datetime(2026, 9, 8, 9, 10)),
        # And an offset that is not UTC.
        (
            datetime(2026, 9, 8, 11, 0, tzinfo=BERLIN),
            datetime(2026, 9, 8, 11, 10, tzinfo=BERLIN),
        ),
    ],
)
def test_both_timestamps_are_utc_with_a_z_suffix(final_state, run_config, started, completed):
    """The wire form is UTC with Z, whatever the caller's clock says."""
    payload = build(final_state, run_config, started_at=started, completed_at=completed)
    assert payload["run_started_at"].endswith("Z")
    assert payload["run_completed_at"].endswith("Z")
    assert "+" not in payload["run_started_at"]


def test_an_offset_timestamp_is_converted_not_relabelled(final_state, run_config):
    payload = build(
        final_state,
        run_config,
        started_at=datetime(2026, 9, 8, 11, 0, tzinfo=BERLIN),
        completed_at=datetime(2026, 9, 8, 11, 10, tzinfo=BERLIN),
    )
    assert payload["run_started_at"] == "2026-09-08T09:00:00Z"
    assert payload["run_completed_at"] == "2026-09-08T09:10:00Z"


def test_unparseable_decision_becomes_review(final_state, run_config):
    final_state["final_trade_decision"] = "I could not decide."
    payload = build(final_state, run_config)
    assert payload["decision"]["rating"] == "REVIEW"
    assert payload["decision"]["executive_summary"] == "I could not decide."
    # The raw text is the summary, and only the summary: a parse failure is not
    # an investment thesis.
    assert payload["decision"]["investment_thesis"] == ""
    assert "trader_proposal" in payload


def test_a_bare_rating_line_is_read_by_the_frameworks_own_extractor():
    """Prose the label parser cannot read is still worth a rating, not REVIEW.

    And the rating keeps the text it was read from: a tradeable rating with an
    empty summary is a recommendation with no reasoning behind it, which is
    exactly what the REVIEW branch refuses to send.
    """
    raw = "Rating: Buy\n\nWe are constructive."
    decision = parse_decision(raw)
    assert decision["rating"] == "Buy"
    assert decision["executive_summary"] == raw
    assert decision["investment_thesis"] == ""


def test_a_bold_heading_inside_a_thesis_does_not_truncate_it():
    markdown = """**Rating**: Buy

**Investment Thesis**: Cloud revenue growth is durable.

**Three drivers**

Backlog, pricing and margin all point the same way.

**Price Target**: 260.0"""
    decision = parse_decision(markdown)
    assert decision["investment_thesis"].endswith("point the same way.")
    assert "Three drivers" in decision["investment_thesis"]
    assert decision["price_target"] == 260.0


def test_a_rating_quoted_inside_a_thesis_does_not_override_the_real_one():
    markdown = """**Rating**: Buy

**Investment Thesis**: The bear case is stated as "**Rating**: Sell" and we disagree."""
    decision = parse_decision(markdown)
    assert decision["rating"] == "Buy"
    assert "Sell" in decision["investment_thesis"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("260.0", 260.0),
        # A lone dot is the decimal point. `str(float)` is what wrote the
        # number, so reading these as thousands groups made a 1.234 target
        # arrive as 1234 and an 0.085 target as 85.
        ("1.234", 1.234),
        ("0.085", 0.085),
        ("12.345", 12.345),
        ("$1,250", 1250.0),
        ("1,234.56", 1234.56),
        ("1.234,50", 1234.5),
        ("EUR 241.5", 241.5),
        ("290", 290.0),
        # A second number in the field means we do not know which one is meant.
        ("We see 12-15% upside to 290", None),
        ("12-15% upside to 290", None),
        # 1.25 or 125? A hundredfold error is worse than no price target.
        ("1,25", None),
        ("", None),
        (None, None),
    ],
)
def test_number_parses_only_unambiguous_single_numbers(raw, expected):
    assert _number(raw) == expected


@pytest.mark.parametrize("target", [1.234, 0.085, 260.0, 1250.0])
def test_a_rendered_price_target_survives_the_round_trip(target):
    """The convention, not the one call site: whatever the framework renders,
    we read back unchanged.

    ``render_pm_decision`` writes the target with ``str(float)``, so this is the
    shape every real price target arrives in. A separator rule that disagrees
    with it does not fail loudly; it stores a plausible number that is wrong by
    a factor of a thousand.
    """
    from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision

    markdown = render_pm_decision(
        PortfolioDecision(
            rating="Overweight",
            executive_summary="Constructive over the next quarter.",
            investment_thesis="Cloud revenue growth is durable.",
            price_target=target,
        )
    )

    assert parse_decision(markdown)["price_target"] == target


def test_missing_trader_plan_omits_the_proposal(final_state, run_config):
    final_state["trader_investment_plan"] = ""
    payload = build(final_state, run_config)
    assert "trader_proposal" not in payload
