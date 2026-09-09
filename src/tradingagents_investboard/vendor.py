"""The framework's core data methods against the Investboard agent-data reads.

Importing this module registers ``investboard`` in the framework's vendor
registry; ``cli.analyze`` imports it for that side effect and points the four
core categories at it. Each method returns the same text shape the framework's
yfinance vendor returns for it, so the agents read one format whichever vendor
served them.

Indicators are computed locally with ``stockstats`` from Investboard OHLCV,
mirroring the framework's yfinance path, so no indicator endpoint exists.

Only what the app already serves comes from here: sentiment, macro and
prediction markets stay on the framework's own vendors, and ``get_global_news``
is not registered, so a ``news_data`` chain of ``investboard,yfinance`` serves
company news from Investboard and global news from yfinance.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
from stockstats import wrap
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS

from .auth import access_token, base_url
from .client import InvestboardApiError, InvestboardClient

VENDOR_NAME = "investboard"

# Test seam: a MockTransport routed through here never reads the token file.
_transport: httpx.BaseTransport | None = None
_client_instance: InvestboardClient | None = None


def _reset_client() -> None:
    global _client_instance
    _client_instance = None


def _client() -> InvestboardClient:
    """One client per process; the token is read when the client is built."""
    global _client_instance
    if _client_instance is None:
        _client_instance = InvestboardClient(base_url(), access_token(), transport=_transport)
    return _client_instance


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _retrieved_on() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _translate(error: InvestboardApiError, symbol: str) -> Exception:
    """The framework's own error taxonomy, so its vendor chain can act on ours.

    A cap or a provider outage is a rate limit to the router (next vendor, or
    wait). A credential fault, a paused account or an instrument outside the
    data scope is a configuration the user has to fix, and the message carries
    the server's hint so the CLI prints the next step. An instrument the
    provider does not carry is no market data.
    """
    if error.status in (429, 503):
        return VendorRateLimitError(f"Investboard: {error.message}")
    if error.status in (401, 402, 403):
        return VendorNotConfiguredError(f"Investboard: {error.reason}: {error.message}")
    if error.status == 422:
        return NoMarketDataError(symbol, None, error.message)
    return error


def _call(symbol: str, fn: Callable[[InvestboardClient], Any]) -> Any:
    try:
        return fn(_client())
    except InvestboardApiError as error:
        raise _translate(error, symbol) from error


# --- OHLCV ---------------------------------------------------------------


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Daily OHLCV as the framework's CSV text; the server renders the shape."""
    text = _call(symbol, lambda c: c.get_ohlcv(symbol, start_date, end_date))
    if "# Total records: 0" in text:
        raise NoMarketDataError(symbol, symbol, f"no rows between {start_date} and {end_date}")
    return text


def _frame_from_csv(text: str) -> pd.DataFrame:
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    frame = pd.read_csv(io.StringIO(body))
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date").sort_index()
    frame.columns = [column.lower() for column in frame.columns]
    return frame


SUPPORTED_INDICATORS = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as "
        "dynamic support/resistance."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify "
        "golden/death cross setups."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and the "
        "short-term trend."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and "
        "divergence as signals of trend changes."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line "
        "to trigger trades."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize "
        "momentum strength and spot divergence early."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 "
        "thresholds and watch for divergence."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a "
        "dynamic benchmark for price movement."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: "
        "Signals potential overbought conditions."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: "
        "Indicates potential oversold conditions."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust "
        "position sizes based on volatility."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price "
        "action with volume data."
    ),
    "mfi": (
        "MFI: The Money Flow Index incorporates volume. Usage: Identify overbought/oversold "
        "conditions with volume confirmation."
    ),
}

# Sessions the longest supported window (the 200 SMA) needs before the first
# reported day, plus a margin for holidays.
_HISTORY_SESSIONS = 320


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    if indicator not in SUPPORTED_INDICATORS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Choose from: {', '.join(SUPPORTED_INDICATORS)}"
        )
    end = date.fromisoformat(curr_date)
    start = end - timedelta(days=int(_HISTORY_SESSIONS * 1.5))
    text = get_stock_data(symbol, start.isoformat(), end.isoformat())
    frame = wrap(_frame_from_csv(text))
    series = frame[indicator]
    by_day = {index.date(): value for index, value in series.items()}
    lines = []
    day = end
    first = end - timedelta(days=look_back_days)
    while day >= first:
        value = by_day.get(day)
        if value is None or pd.isna(value):
            lines.append(f"{day.isoformat()}: N/A: Not a trading day (weekend or holiday)")
        else:
            lines.append(f"{day.isoformat()}: {float(value):.4f}")
        day -= timedelta(days=1)
    header = f"## {indicator} values from {first.isoformat()} to {curr_date}:"
    return f"{header}\n\n" + "\n".join(lines) + f"\n\n{SUPPORTED_INDICATORS[indicator]}"


# --- Fundamentals and statements ------------------------------------------

_FUNDAMENTAL_LABELS: tuple[tuple[str, str, str], ...] = (
    # (label, block, field)
    ("Name", "profile", "companyName"),
    ("Sector", "profile", "sector"),
    ("Industry", "profile", "industry"),
    ("Market Cap", "profile", "marketCap"),
    ("Beta", "profile", "beta"),
    ("PE Ratio (TTM)", "ratios_ttm", "priceToEarningsRatioTTM"),
    ("Dividend Yield", "ratios_ttm", "dividendYieldTTM"),
    ("Profit Margin", "ratios_ttm", "netProfitMarginTTM"),
    ("Operating Margin", "ratios_ttm", "operatingProfitMarginTTM"),
    ("PEG Ratio", "ratios_ttm", "priceToEarningsGrowthRatioTTM"),
    ("Price to Book", "key_metrics", "pbRatio"),
    ("EPS", "key_metrics", "eps"),
    ("Return on Equity", "key_metrics", "roe"),
    ("Return on Assets", "key_metrics", "returnOnAssets"),
    ("Debt to Equity", "key_metrics", "debtToEquity"),
    ("Current Ratio", "key_metrics", "currentRatio"),
    ("Free Cash Flow per Share", "key_metrics", "freeCashFlowPerShare"),
)

# The blocks the server carries undated, so an `asOf` read cannot bound them.
_CURRENT_VALUE_BLOCKS = ("profile", "ratios_ttm")


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    data = _call(ticker, lambda c: c.get_fundamentals(ticker, curr_date))
    point_in_time = data.get("point_in_time") or {}
    blocks = {
        "profile": data.get("profile") or {},
        "ratios_ttm": data.get("ratios_ttm") or {},
        "key_metrics": (data.get("key_metrics") or [{}])[0],
    }
    lines = [
        f"# Company Fundamentals for {ticker}",
        f"# Data retrieved on: {_retrieved_on()}",
        "",
    ]
    for label, block, field in _FUNDAMENTAL_LABELS:
        value = blocks[block].get(field)
        if value is None:
            continue
        note = ""
        # Say so where the number is today's, so the agent never reads a
        # current value as the value on the analysis date.
        if curr_date and block in _CURRENT_VALUE_BLOCKS and not point_in_time.get(block, False):
            note = f" (current values, not as of {curr_date})"
        lines.append(f"{label}: {value}{note}")
    if len(lines) == 3:
        raise NoMarketDataError(ticker, ticker, "no fundamental fields returned")
    return "\n".join(lines)


_STATEMENT_HEADERS = {
    "income": "Income Statement",
    "balance": "Balance Sheet",
    "cashflow": "Cash Flow",
}


def _statement(ticker: str, statement: str, freq: str, curr_date: str | None) -> str:
    period = "quarterly" if (freq or "").lower() == "quarterly" else "annual"
    data = _call(ticker, lambda c: c.get_statements(ticker, statement, period, curr_date))
    rows = data.get("rows") or []
    if not rows:
        raise NoMarketDataError(ticker, ticker, f"no {_STATEMENT_HEADERS[statement].lower()} data")
    frame = pd.DataFrame(rows).set_index("date").drop(columns=["period"], errors="ignore").T
    frame.columns = [str(column) for column in frame.columns]
    header = (
        f"# {_STATEMENT_HEADERS[statement]} data for {ticker} ({period})\n"
        f"# Data retrieved on: {_retrieved_on()}\n\n"
    )
    return header + frame.to_csv()


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, "balance", freq, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, "cashflow", freq, curr_date)


def get_income_statement(
    ticker: str, freq: str = "quarterly", curr_date: str | None = None
) -> str:
    return _statement(ticker, "income", freq, curr_date)


# --- News and insider transactions ----------------------------------------


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    data = _call(ticker, lambda c: c.get_news(ticker, start_date, end_date))
    items = data.get("items") or []
    if not items:
        return f"No news found for {ticker} between {start_date} and {end_date}"
    out = [f"## {ticker} News, from {start_date} to {end_date}:", ""]
    for item in items:
        out.append(f"### {item['title']} (source: {item.get('publisher') or 'unknown'})")
        if item.get("summary"):
            out.append(item["summary"])
        if item.get("url"):
            out.append(f"Link: {item['url']}")
        out.append("")
    return "\n".join(out) + "\n"


_INSIDER_COLUMNS = ("date", "name", "role", "action", "transaction_type", "shares", "price")


def get_insider_transactions(ticker: str) -> str:
    end = _today()
    start = (date.fromisoformat(end) - timedelta(days=365)).isoformat()
    data = _call(ticker, lambda c: c.get_insider(ticker, start, end))
    items = data.get("items") or []
    if not items:
        return f"No insider transactions reported for symbol '{ticker}'"
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_INSIDER_COLUMNS)
    for item in items:
        writer.writerow(
            [
                str(item.get("date", ""))[:10],
                *(
                    item.get(column) if item.get(column) is not None else ""
                    for column in _INSIDER_COLUMNS[1:]
                ),
            ]
        )
    return (
        f"# Insider Transactions data for {ticker}\n"
        f"# Data retrieved on: {_retrieved_on()}\n\n{buffer.getvalue()}"
    )


# --- Registration -----------------------------------------------------------

_METHODS: dict[str, Any] = {
    "get_stock_data": get_stock_data,
    "get_indicators": get_indicators,
    "get_fundamentals": get_fundamentals,
    "get_balance_sheet": get_balance_sheet,
    "get_cashflow": get_cashflow,
    "get_income_statement": get_income_statement,
    "get_news": get_news,
    "get_insider_transactions": get_insider_transactions,
}

for _method, _impl in _METHODS.items():
    VENDOR_METHODS.setdefault(_method, {})[VENDOR_NAME] = _impl
if VENDOR_NAME not in VENDOR_LIST:
    VENDOR_LIST.append(VENDOR_NAME)
