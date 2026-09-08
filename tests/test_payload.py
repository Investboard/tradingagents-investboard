from datetime import datetime, timezone

from tradingagents_investboard.payload import build_run_payload, run_id_for

STARTED = datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc)
ENDED = datetime(2026, 9, 8, 9, 10, tzinfo=timezone.utc)


def test_run_id_is_deterministic_per_ticker_date_and_start():
    a = run_id_for("SAP.DE", "2026-09-08", STARTED)
    b = run_id_for("SAP.DE", "2026-09-08", STARTED)
    c = run_id_for("SAP.DE", "2026-09-08", ENDED)
    assert a == b
    assert a != c
    assert a.startswith("SAP.DE_2026-09-08_")


def test_payload_reads_the_rendered_decision_and_proposal(final_state, run_config):
    payload = build_run_payload(
        final_state=final_state,
        ticker="SAP.DE",
        trade_date="2026-09-08",
        config=run_config,
        started_at=STARTED,
        completed_at=ENDED,
        framework_version="0.4.1",
    )
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
    assert payload["run_started_at"] == "2026-09-08T09:00:00+00:00"


def test_unparseable_decision_becomes_review(final_state, run_config):
    final_state["final_trade_decision"] = "I could not decide."
    payload = build_run_payload(
        final_state=final_state,
        ticker="SAP.DE",
        trade_date="2026-09-08",
        config=run_config,
        started_at=STARTED,
        completed_at=ENDED,
        framework_version="0.4.1",
    )
    assert payload["decision"]["rating"] == "REVIEW"
    assert payload["decision"]["executive_summary"] == "I could not decide."
    assert "trader_proposal" in payload


def test_missing_trader_plan_omits_the_proposal(final_state, run_config):
    final_state["trader_investment_plan"] = ""
    payload = build_run_payload(
        final_state=final_state,
        ticker="SAP.DE",
        trade_date="2026-09-08",
        config=run_config,
        started_at=STARTED,
        completed_at=ENDED,
        framework_version="0.4.1",
    )
    assert "trader_proposal" not in payload
