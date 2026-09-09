from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS

from tradingagents_investboard import vendor

OHLCV_CSV = (
    "# Stock data for SAP.DE from 2026-09-01 to 2026-09-03\n"
    "# Total records: 3\n"
    "# Data retrieved on: 2026-09-09 10:00:00\n"
    "\n"
    "Date,Open,High,Low,Close,Volume\n"
    "2026-09-01,100,101,99,100.5,1000\n"
    "2026-09-02,101,102,100,101.5,1100\n"
    "2026-09-03,102,103,101,102.5,\n"
)


def envelope(data: object) -> httpx.Response:
    return httpx.Response(200, json={"data": data, "error": None, "meta": {}})


def refusal(status: int, reason: str, hint: str | None = None) -> httpx.Response:
    details = {"reason": reason}
    if hint:
        details["hint"] = hint
    return httpx.Response(
        status,
        json={
            "data": None,
            "error": {"code": "X", "message": reason, "details": details},
            "meta": {},
        },
    )


@pytest.fixture
def served(monkeypatch):
    """Route every vendor call to a handler; the vendor never reads the token file here."""
    calls: list[httpx.Request] = []

    def install(handler):
        def routed(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return handler(request)

        monkeypatch.setattr(vendor, "_transport", httpx.MockTransport(routed))
        monkeypatch.setattr(vendor, "access_token", lambda *a, **k: "tok")
        monkeypatch.setattr(vendor, "base_url", lambda: "https://app.example")
        vendor._reset_client()
        return calls

    return install


def test_importing_the_module_registers_the_eight_methods_and_nothing_else():
    assert "investboard" in VENDOR_LIST
    registered = {m for m, impls in VENDOR_METHODS.items() if "investboard" in impls}
    assert registered == {
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
        "get_insider_transactions",
    }


def test_stock_data_hands_the_server_csv_through_unchanged(served):
    served(lambda r: httpx.Response(200, text=OHLCV_CSV, headers={"content-type": "text/csv"}))
    assert vendor.get_stock_data("SAP.DE", "2026-09-01", "2026-09-03") == OHLCV_CSV


def test_an_empty_window_is_no_market_data(served):
    empty = (
        OHLCV_CSV.split("\n\n")[0].replace("Total records: 3", "Total records: 0")
        + "\n\nDate,Open,High,Low,Close,Volume\n"
    )
    served(lambda r: httpx.Response(200, text=empty, headers={"content-type": "text/csv"}))
    with pytest.raises(NoMarketDataError):
        vendor.get_stock_data("SAP.DE", "2026-09-01", "2026-09-03")


def test_indicators_are_computed_locally_in_the_framework_layout(served):
    # Trading days only, so the September weekend is absent from the server's
    # rows and reads as the framework's "not a trading day" line.
    first, last = date(2026, 8, 1), date(2026, 9, 8)
    days = [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
        if (first + timedelta(days=offset)).weekday() < 5
    ]
    rows = "\n".join(
        f"{day},{100 + i},{101 + i},{99 + i},{100.5 + i},{1000}" for i, day in enumerate(days)
    )
    csv = (
        f"# Stock data\n# Total records: {len(days)}\n# Data retrieved on: x\n\n"
        f"Date,Open,High,Low,Close,Volume\n{rows}\n"
    )
    calls = served(lambda r: httpx.Response(200, text=csv, headers={"content-type": "text/csv"}))
    text = vendor.get_indicators("SAP.DE", "close_10_ema", "2026-09-08", 3)
    assert text.startswith("## close_10_ema values from 2026-09-05 to 2026-09-08:\n\n")
    assert "2026-09-08: " in text
    assert "2026-09-06: N/A: Not a trading day (weekend or holiday)" in text
    assert text.rstrip().endswith("short-term trend.")
    assert calls[0].url.params["to"] == "2026-09-08"
    with pytest.raises(ValueError):
        vendor.get_indicators("SAP.DE", "not_an_indicator", "2026-09-08", 3)


def test_fundamentals_render_the_label_lines(served):
    served(
        lambda r: envelope(
            {
                "subject": {"ticker": "SAP.DE", "name": "SAP SE"},
                "as_of": "2026-09-08",
                "point_in_time": {"profile": False, "key_metrics": True, "ratios_ttm": False},
                "profile": {
                    "companyName": "SAP SE",
                    "sector": "Technology",
                    "industry": "Software",
                    "marketCap": 250000000000,
                    "beta": 1.1,
                },
                "key_metrics": [
                    {
                        "date": "2025-12-31",
                        "pbRatio": 5.1,
                        "roe": 0.12,
                        "debtToEquity": 0.2,
                        "eps": 4.5,
                        "currentRatio": 1.3,
                    }
                ],
                "ratios_ttm": {
                    "priceToEarningsRatioTTM": 30.5,
                    "dividendYieldTTM": 0.011,
                    "netProfitMarginTTM": 0.19,
                },
            }
        )
    )
    text = vendor.get_fundamentals("SAP.DE", "2026-09-08")
    assert text.startswith("# Company Fundamentals for SAP.DE\n")
    assert "Name: SAP SE" in text and "Sector: Technology" in text
    assert "PE Ratio (TTM): 30.5" in text and "Dividend Yield: 0.011" in text
    assert "Price to Book: 5.1" in text and "Return on Equity: 0.12" in text
    assert "(current values, not as of 2026-09-08)" in text


def test_statements_render_as_line_items_by_period(served):
    served(
        lambda r: envelope(
            {
                "subject": {"ticker": "SAP.DE"},
                "statement": r.url.params["statement"],
                "period": r.url.params["period"],
                "as_of": "2026-09-08",
                "rows": [
                    {"date": "2026-06-30", "period": "Q2", "revenue": 9.0, "netIncome": 1.5},
                    {"date": "2026-03-31", "period": "Q1", "revenue": 8.5, "netIncome": 1.2},
                ],
            }
        )
    )
    text = vendor.get_income_statement("SAP.DE", "quarterly", "2026-09-08")
    assert text.startswith("# Income Statement data for SAP.DE (quarterly)\n")
    lines = text.split("\n\n", 1)[1].splitlines()
    assert lines[0] == ",2026-06-30,2026-03-31"
    assert "revenue,9.0,8.5" in lines
    assert vendor.get_balance_sheet("SAP.DE", "annual", "2026-09-08").startswith(
        "# Balance Sheet data"
    )
    assert vendor.get_cashflow("SAP.DE", "annual", "2026-09-08").startswith("# Cash Flow data")


def test_news_renders_the_framework_blocks_and_the_empty_sentence(served):
    served(
        lambda r: envelope(
            {
                "subject": {"ticker": "SAP.DE"},
                "from": "2026-09-01",
                "to": "2026-09-08",
                "items": [
                    {
                        "published_at": "2026-09-02T08:00:00.000Z",
                        "title": "Cloud beat",
                        "publisher": "Reuters",
                        "summary": "Strong quarter",
                        "url": "https://r.example/a",
                    }
                ],
            }
        )
    )
    text = vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert text == (
        "## SAP.DE News, from 2026-09-01 to 2026-09-08:\n\n"
        "### Cloud beat (source: Reuters)\nStrong quarter\nLink: https://r.example/a\n\n"
    )
    served(
        lambda r: envelope(
            {"subject": {"ticker": "SAP.DE"}, "from": "2026-09-01", "to": "2026-09-08", "items": []}
        )
    )
    assert vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08") == (
        "No news found for SAP.DE between 2026-09-01 and 2026-09-08"
    )


def test_insider_transactions_render_as_csv_or_the_empty_sentence(served, monkeypatch):
    monkeypatch.setattr(vendor, "_today", lambda: "2026-09-09")
    calls = served(
        lambda r: envelope(
            {
                "subject": {"ticker": "SAP.DE"},
                "from": "2025-09-09",
                "to": "2026-09-08",
                "items": [
                    {
                        "date": "2026-09-02T00:00:00.000Z",
                        "name": "A",
                        "role": "CEO",
                        "action": "buy",
                        "transaction_type": "P-Purchase",
                        "shares": 100,
                        "price": 120.0,
                    }
                ],
            }
        )
    )
    text = vendor.get_insider_transactions("SAP.DE")
    assert text.startswith("# Insider Transactions data for SAP.DE\n")
    assert "date,name,role,action,transaction_type,shares,price" in text
    assert "2026-09-02,A,CEO,buy,P-Purchase,100,120.0" in text
    assert calls[0].url.params["from"] == "2025-09-09"
    served(
        lambda r: envelope({"subject": {"ticker": "SAP.DE"}, "from": "x", "to": "y", "items": []})
    )
    assert vendor.get_insider_transactions("SAP.DE") == (
        "No insider transactions reported for symbol 'SAP.DE'"
    )


@pytest.mark.parametrize(
    "status,reason,expected",
    [
        (429, "daily_read_cap_reached", VendorRateLimitError),
        (503, "provider_unavailable", VendorRateLimitError),
        (401, "unauthorized", VendorNotConfiguredError),
        (402, "access_paused", VendorNotConfiguredError),
        (403, "subject_out_of_scope", VendorNotConfiguredError),
        (422, "subject_unresolvable", NoMarketDataError),
    ],
)
def test_refusals_map_to_the_framework_errors(served, status, reason, expected):
    served(lambda r: refusal(status, reason, "register it"))
    with pytest.raises(expected) as raised:
        vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    if reason == "subject_out_of_scope":
        assert "register it" in str(raised.value)
