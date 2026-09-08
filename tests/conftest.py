import pytest

PM_DECISION = """**Rating**: Overweight

**Executive Summary**: Constructive view over the next quarter.

**Investment Thesis**: Cloud revenue growth is durable.

**Price Target**: 260.0

**Time Horizon**: 3-6 months"""

TRADER_PLAN = """**Action**: Buy

**Reasoning**: Momentum and fundamentals agree.

**Entry Price**: 241.5

**Stop Loss**: 225.0

**Position Sizing**: 3% of portfolio

FINAL TRANSACTION PROPOSAL: **BUY**"""


@pytest.fixture
def final_state() -> dict:
    return {
        "company_of_interest": "SAP.DE",
        "trade_date": "2026-09-08",
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "bull_history": "Bull Analyst: up",
            "bear_history": "Bear Analyst: down",
            "judge_decision": "**Recommendation**: Overweight",
        },
        "trader_investment_plan": TRADER_PLAN,
        "risk_debate_state": {
            "aggressive_history": "Aggressive Analyst: go",
            "conservative_history": "Conservative Analyst: careful",
            "neutral_history": "Neutral Analyst: balance",
            "judge_decision": PM_DECISION,
        },
        "final_trade_decision": PM_DECISION,
    }


@pytest.fixture
def run_config() -> dict:
    return {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-opus-5",
        "quick_think_llm": "claude-sonnet-5",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "output_language": "English",
    }
