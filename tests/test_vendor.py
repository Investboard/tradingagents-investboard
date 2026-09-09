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

from tradingagents_investboard import _session, vendor
from tradingagents_investboard.auth import NOT_CONNECTED
from tradingagents_investboard.client import InvestboardApiError

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


def refusal(
    status: int,
    reason: str,
    hint: str | None = None,
    message: str = "the server refused the read",
) -> httpx.Response:
    """A refusal whose ``message`` is not its ``reason``, so a test can tell them apart."""
    details = {"reason": reason}
    if hint:
        details["hint"] = hint
    return httpx.Response(
        status,
        json={
            "data": None,
            "error": {"code": "X", "message": message, "details": details},
            "meta": {},
        },
    )


def trading_days(last: date, count: int) -> list[date]:
    """``count`` weekdays ending on ``last``, oldest first."""
    days: list[date] = []
    day = last
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return sorted(days)


def sessions_csv(
    days: list[date],
    volume: str = "1000",
    truncated: str | None = None,
    base: float = 100.0,
    volumes: dict[int, str] | None = None,
) -> str:
    """The server's OHLCV CSV over ``days``, one row a session.

    ``volumes`` overrides the volume of individual rows by index, which is how
    a single gap among valid rows is expressed.
    """
    rows = "\n".join(
        f"{day.isoformat()},{base + i},{base + 1 + i},{base - 1 + i},{base + 0.5 + i},"
        f"{(volumes or {}).get(i, volume)}"
        for i, day in enumerate(days)
    )
    header = ["# Stock data", f"# Total records: {len(days)}"]
    if truncated is not None:
        header.append(f"# Truncated: {truncated}")
    header.append("# Data retrieved on: x")
    return "\n".join(header) + "\n\nDate,Open,High,Low,Close,Volume\n" + rows + "\n"


def csv_response(text: str) -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/csv"})


@pytest.fixture
def served(monkeypatch):
    """Route every vendor call to a handler; the vendor never reads the token file here."""
    calls: list[httpx.Request] = []

    def install(handler):
        def routed(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return handler(request)

        monkeypatch.setattr(_session, "_transport", httpx.MockTransport(routed))
        monkeypatch.setattr(_session, "access_token", lambda *a, **k: "tok")
        monkeypatch.setattr(_session, "base_url", lambda: "https://app.example")
        _session._reset_client()
        return calls

    yield install
    # The client is a module singleton: left in place it would carry this
    # test's transport into the next one.
    _session._reset_client()


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
                    # Older first, so a reader that takes position 0 as the
                    # latest period dates every metric to the wrong filing.
                    {"date": "2024-12-31", "pbRatio": 9.9, "roe": 0.03, "eps": 1.1},
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
    assert "Key metrics as of: 2025-12-31" in text
    assert "(current values, not as of 2026-09-08)" in text


def fundamentals(key_metrics: list[dict], point_in_time: dict) -> object:
    return {
        "subject": {"ticker": "SAP.DE"},
        "as_of": "2026-09-08",
        "point_in_time": point_in_time,
        "profile": {"companyName": "SAP SE"},
        "key_metrics": key_metrics,
        "ratios_ttm": {"priceToEarningsRatioTTM": 30.5},
    }


def test_key_metrics_dated_after_the_analysis_date_are_not_the_latest(served):
    # A filing the analysis date could not have seen is look-ahead bias, and
    # `max` over the list picks it deterministically every run.
    served(
        lambda r: envelope(
            fundamentals(
                [
                    {"date": "2025-12-31", "eps": 4.5},
                    {"date": "2026-12-31", "eps": 9.9},
                ],
                {"profile": True, "key_metrics": True, "ratios_ttm": True},
            )
        )
    )
    text = vendor.get_fundamentals("SAP.DE", "2026-09-08")
    assert "Key metrics as of: 2025-12-31" in text
    assert "EPS: 4.5" in text
    assert "2026-12-31" not in text


def test_key_metrics_the_server_did_not_bound_carry_the_current_values_note(served):
    served(
        lambda r: envelope(
            fundamentals(
                [{"date": "2025-12-31", "eps": 4.5}],
                {"profile": True, "key_metrics": False, "ratios_ttm": True},
            )
        )
    )
    text = vendor.get_fundamentals("SAP.DE", "2026-09-08")
    assert "EPS: 4.5 (current values, not as of 2026-09-08)" in text
    # The blocks the server did bound say nothing.
    assert "Name: SAP SE\n" in text + "\n"
    assert "PE Ratio (TTM): 30.5\n" in text + "\n"


def test_an_undated_analysis_bounds_nothing_and_notes_nothing(served):
    served(
        lambda r: envelope(
            fundamentals([{"date": "2026-12-31", "eps": 9.9}], {"key_metrics": False})
        )
    )
    text = vendor.get_fundamentals("SAP.DE")
    assert "Key metrics as of: 2026-12-31" in text
    assert "current values" not in text


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
    assert "the server refused the read" in str(raised.value)
    if expected is VendorNotConfiguredError:
        # The reason is the word the CLI turns into a next step, and it is not
        # the message: it survives only if this branch names it.
        assert reason in str(raised.value)
    if reason == "subject_out_of_scope":
        assert "register it" in str(raised.value)


def test_a_traded_day_without_a_computed_value_reads_na_not_a_holiday(served):
    # Three sessions, so the first has no standard deviation behind it and
    # `boll_ub` cannot be computed on it. That day traded, so the framework's
    # holiday sentence would be a lie about the market; "N/A" is the truth.
    days = trading_days(date(2026, 9, 8), 3)
    served(lambda r: csv_response(sessions_csv(days)))
    lines = vendor.get_indicators("SAP.DE", "boll_ub", "2026-09-08", 6).splitlines()
    assert "2026-09-04: N/A" in lines
    assert "2026-09-04: N/A: Not a trading day (weekend or holiday)" not in lines
    assert "2026-09-05: N/A: Not a trading day (weekend or holiday)" in lines


def test_a_frame_too_short_for_the_200_sma_reports_the_gap_not_a_plausible_number(served):
    # Sixty sessions, far short of the 200 the average needs. stockstats fills
    # a moving average from the first row it holds, so without the warm-up mask
    # these render as numbers a reader cannot tell from a real 200 SMA, and the
    # framework's own yfinance path would answer differently for the same day.
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days)))
    lines = vendor.get_indicators("SAP.DE", "close_200_sma", "2026-09-08", 5).splitlines()
    traded = [line for line in lines if line[:10] in {"2026-09-04", "2026-09-07", "2026-09-08"}]
    # The render walks the window backwards, newest day first.
    assert traded == ["2026-09-08: N/A", "2026-09-07: N/A", "2026-09-04: N/A"]
    # A day that traded is still not a market closure.
    assert "2026-09-05: N/A: Not a trading day (weekend or holiday)" in lines


def test_the_first_row_of_the_frame_is_never_a_reported_value(served):
    # MACD has no single window to warm up, so the mask cannot be sized from
    # the name. Row 0 is still seeded by one session: stockstats reports 0.0
    # there, which reads as a settled momentum reading that nothing supports.
    days = trading_days(date(2026, 9, 8), 3)
    served(lambda r: csv_response(sessions_csv(days)))
    lines = vendor.get_indicators("SAP.DE", "macd", "2026-09-08", 6).splitlines()
    assert "2026-09-04: N/A" in lines
    assert not any(line.startswith("2026-09-04: 0") for line in lines)


def test_a_derived_window_shorter_than_the_frame_still_reports_its_days(served):
    # The mask must blank the warm-up and nothing beyond it: a 10 EMA over
    # sixty sessions has every reported day covered.
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days)))
    lines = vendor.get_indicators("SAP.DE", "close_10_ema", "2026-09-08", 5).splitlines()
    traded = [line for line in lines if line[:10] in {"2026-09-04", "2026-09-07", "2026-09-08"}]
    assert all(": N/A" not in line for line in traded)


def test_one_null_volume_among_valid_rows_is_absorbed_by_the_rolling_sums(served):
    # A characterisation, not an endorsement: stockstats sums volume with
    # `min_periods=1` (stockstats.py:1130), and a pandas rolling sum skips a
    # NaN rather than propagating it, so a single missing volume leaves the
    # VWMA of that day and the thirteen after it computed from a short window
    # and still printed as a number. Only an entirely null volume column
    # reaches the N/A branch. The warm-up mask does not reach this class.
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days, volumes={59: ""})))
    lines = vendor.get_indicators("SAP.DE", "vwma", "2026-09-08", 5).splitlines()
    assert "2026-09-08: 152.1666667" in lines
    # The same day with every volume present. The gap moves the number rather
    # than removing it, which is the case this fixture exists to pin.
    served(lambda r: csv_response(sessions_csv(days)))
    lines = vendor.get_indicators("SAP.DE", "vwma", "2026-09-08", 5).splitlines()
    assert "2026-09-08: 152.6666667" in lines


def test_a_value_above_a_million_renders_decimal_not_scientific(served):
    # The framework's yfinance path renders `str(value)`; an exponent here
    # would read as a different number to an agent comparing the two.
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days, base=1234567.0)))
    text = vendor.get_indicators("SAP.DE", "close_10_ema", "2026-09-08", 3)
    # `.6g` renders this same value as "1.23462e+06".
    assert "2026-09-08: 1234622" in text
    assert "e+" not in text


def test_volume_the_server_left_null_reads_na_on_the_days_that_traded(served):
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days, volume="")))
    lines = vendor.get_indicators("SAP.DE", "vwma", "2026-09-08", 5).splitlines()
    assert "2026-09-08: N/A" in lines and "2026-09-07: N/A" in lines
    assert "2026-09-05: N/A: Not a trading day (weekend or holiday)" in lines


@pytest.mark.parametrize("indicator", sorted(vendor.SUPPORTED_INDICATORS))
def test_every_supported_indicator_renders_over_one_frame(served, indicator):
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days)))
    text = vendor.get_indicators("SAP.DE", indicator, "2026-09-08", 3)
    assert text.startswith(f"## {indicator} values from 2026-09-05 to 2026-09-08:\n\n")
    assert text.rstrip().endswith(vendor.SUPPORTED_INDICATORS[indicator])


def test_a_truncated_history_is_declared_above_the_description(served):
    days = trading_days(date(2026, 9, 8), 60)
    served(lambda r: csv_response(sessions_csv(days, truncated="true")))
    text = vendor.get_indicators("SAP.DE", "close_50_sma", "2026-09-08", 3)
    assert vendor.TRUNCATION_NOTE in text
    assert text.rstrip().endswith(vendor.SUPPORTED_INDICATORS["close_50_sma"])
    served(lambda r: csv_response(sessions_csv(days, truncated="false")))
    assert vendor.TRUNCATION_NOTE not in vendor.get_indicators(
        "SAP.DE", "close_50_sma", "2026-09-08", 3
    )


def test_the_window_covers_the_warm_up(served):
    days = trading_days(date(2026, 9, 8), 60)
    calls = served(lambda r: csv_response(sessions_csv(days)))
    vendor.get_indicators("SAP.DE", "close_200_sma", "2026-09-08", 30)
    requested = date.fromisoformat(calls[0].url.params["from"])
    assert (date(2026, 9, 8) - requested).days == 30 + vendor.WARMUP_CALENDAR_DAYS


def test_a_look_back_longer_than_one_read_is_clamped_not_refused(served):
    # The look-back is whatever the model asked for, and a year is a normal
    # ask of a 200 SMA. Raising hands the indicator to the next vendor in the
    # chain, so one run would mix two price sources; the clamp keeps it here
    # and says so in the output.
    days = trading_days(date(2026, 9, 8), 60)
    calls = served(lambda r: csv_response(sessions_csv(days)))
    text = vendor.get_indicators("SAP.DE", "close_50_sma", "2026-09-08", 365)
    clamped = date(2026, 9, 8) - timedelta(days=vendor.MAX_LOOK_BACK_DAYS)
    assert text.startswith(f"## close_50_sma values from {clamped.isoformat()} to 2026-09-08:")
    assert vendor.CLAMP_NOTE.format(asked=365, days=vendor.MAX_LOOK_BACK_DAYS) in text
    requested = date.fromisoformat(calls[0].url.params["from"])
    assert (date(2026, 9, 8) - requested).days == (
        vendor.MAX_LOOK_BACK_DAYS + vendor.WARMUP_CALENDAR_DAYS
    )
    assert text.rstrip().endswith(vendor.SUPPORTED_INDICATORS["close_50_sma"])
    # A look-back inside the cap says nothing about a clamp.
    assert "clamped" not in vendor.get_indicators("SAP.DE", "close_50_sma", "2026-09-08", 30)


def test_a_crlf_body_still_finds_the_header_block(served):
    # The header and the rows are separated by a blank line, which on a CRLF
    # body is "\r\n\r\n". Split on "\n\n" alone it is never found, the header
    # block reads empty and the truncation the server declared is never said.
    days = trading_days(date(2026, 9, 8), 60)
    body = sessions_csv(days, truncated="true").replace("\n", "\r\n")
    served(lambda r: csv_response(body))
    text = vendor.get_indicators("SAP.DE", "close_50_sma", "2026-09-08", 3)
    assert vendor.TRUNCATION_NOTE in text
    assert "2026-09-08: " in text


def test_a_body_that_is_not_the_csv_never_reaches_the_agents(served):
    # What a deployment behind an access wall answers a data read with.
    served(lambda r: httpx.Response(200, text="<!doctype html><title>Sign in</title>"))
    with pytest.raises(InvestboardApiError) as raised:
        vendor.get_stock_data("SAP.DE", "2026-09-01", "2026-09-03")
    assert raised.value.reason == "unexpected_body"


def test_the_client_is_rebuilt_when_the_token_has_been_refreshed(served, monkeypatch):
    calls = served(
        lambda r: envelope({"subject": {"ticker": "SAP.DE"}, "from": "x", "to": "y", "items": []})
    )
    issued = iter(["first", "second"])
    monkeypatch.setattr(_session, "access_token", lambda *a, **k: next(issued, "second"))
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert [call.headers["authorization"] for call in calls] == ["Bearer first", "Bearer second"]


def test_the_token_is_read_under_the_lock_that_guards_the_client(served, monkeypatch):
    # Read outside it, two threads interleave and the loser rebuilds the
    # singleton with the token it read before the refresh, so the next request
    # carries a dead bearer and the run silently changes vendor. Reading it
    # under the lock also means one refresh exchange rather than one a thread.
    held: list[bool] = []
    served(
        lambda r: envelope({"subject": {"ticker": "SAP.DE"}, "from": "x", "to": "y", "items": []})
    )

    def token(*args, **kwargs):
        held.append(_session._client_lock.locked())
        return "tok"

    monkeypatch.setattr(_session, "access_token", token)
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert held == [True]


def test_a_superseded_client_is_closed_once_nothing_can_still_be_serving_it(served, monkeypatch):
    served(
        lambda r: envelope({"subject": {"ticker": "SAP.DE"}, "from": "x", "to": "y", "items": []})
    )
    issued = iter(["first", "second", "third"])
    monkeypatch.setattr(_session, "access_token", lambda *a, **k: next(issued, "third"))
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    first = _session._client_instance
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    second = _session._client_instance
    # Still open: another thread may have taken it before the rotation and be
    # mid-request on it.
    assert second is not first and not first.is_closed
    vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert first.is_closed and not second.is_closed
    _session._reset_client()
    assert second.is_closed


def test_a_refusal_without_details_still_names_the_reason(served):
    served(
        lambda r: httpx.Response(
            402,
            json={"data": None, "error": {"code": "X", "message": "Access is paused"}, "meta": {}},
        )
    )
    with pytest.raises(VendorNotConfiguredError) as raised:
        vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert "access_paused" in str(raised.value)
    assert "Access is paused" in str(raised.value)


def test_an_unreachable_server_and_a_missing_connection_map_to_the_framework_errors(
    served, monkeypatch
):
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    served(unreachable)
    with pytest.raises(VendorRateLimitError):
        vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")

    def not_connected(*args, **kwargs):
        raise RuntimeError(NOT_CONNECTED)

    monkeypatch.setattr(_session, "access_token", not_connected)
    with pytest.raises(VendorNotConfiguredError) as raised:
        vendor.get_news("SAP.DE", "2026-09-01", "2026-09-08")
    assert "tradingagents-investboard connect" in str(raised.value)
