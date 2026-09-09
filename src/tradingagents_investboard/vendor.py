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

The HTTP client, the token it carries and the translation of a refusal into the
framework's errors live in ``_session``; what follows is rendering.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from stockstats import dft_windows, wrap
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS

from ._session import _call

VENDOR_NAME = "investboard"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _retrieved_on() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# --- OHLCV ---------------------------------------------------------------


def _split_csv(text: str) -> tuple[str, str]:
    """The header block and the rows, which the blank line separates.

    Line endings are normalised first: the separator on a CRLF body is
    ``\r\n\r\n``, which does not contain ``\n\n`` at all, so splitting on that
    alone would hand back an empty header block and silence the truncation the
    server declared in it.
    """
    normalised = text.replace("\r\n", "\n")
    header, separator, body = normalised.partition("\n\n")
    return (header, body) if separator else ("", normalised)


def _parse_csv_header(text: str) -> dict[str, str]:
    """The ``# key: value`` lines the server writes above the rows.

    Read as fields rather than searched for as substrings: ``Truncated`` says
    something the reader has to act on, and a substring test cannot tell a
    header line from a value that happens to contain it. Nothing reads the
    record count the server also writes here; emptiness is decided by the rows.
    """
    fields: dict[str, str] = {}
    for line in _split_csv(text)[0].splitlines():
        if not line.startswith("#"):
            continue
        key, separator, value = line.lstrip("#").partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _data_rows(text: str) -> list[str]:
    """The session rows under the column line; empty when the window has none."""
    lines = [line for line in _split_csv(text)[1].splitlines() if line.strip()]
    return lines[1:]


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Daily OHLCV as the framework's CSV text; the server renders the shape."""
    text = _call(symbol, lambda c: c.get_ohlcv(symbol, start_date, end_date))
    # Emptiness is what the rows say, not what the header claims: a count that
    # disagrees with the body would otherwise decide it.
    if not _data_rows(text):
        raise NoMarketDataError(symbol, symbol, f"no rows between {start_date} and {end_date}")
    return text


def _frame_from_csv(text: str) -> pd.DataFrame:
    body = _split_csv(text)[1]
    frame = pd.read_csv(io.StringIO(body))
    # `utc=True` because the window always spans a DST change: offset-bearing
    # server timestamps would otherwise be mixed offsets, which pandas refuses
    # with a ValueError that no translation here would catch.
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
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

# One read is capped at 400 sessions. The window is the reported days plus a
# warm-up ahead of the first of them: 275 + 300 = 575 calendar days. The margins
# are thin, not comfortable, and they are stated here rather than assumed. On a
# market printing about 252 sessions a year, 575 calendar days is roughly 395
# sessions against the 400-row cap, and the 300 warm-up days are roughly 207
# sessions against the 200 the longest supported average needs. A market
# printing fewer than about 243 sessions a year runs the warm-up short, and an
# instrument that trades every day of the week overruns the cap, so the server
# drops its oldest rows, which are the warm-up. Neither case is refused: the
# warm-up is blanked instead (`_with_warmup_blanked`), so a shortfall reads as
# the N/A it is rather than as a number computed from too little history.
WARMUP_CALENDAR_DAYS = 300
# The longest look-back one read can warm up. A longer one is clamped to it,
# not refused: the look-back is model-supplied, and a `ValueError` here is
# caught by the framework's router as a vendor failure, which hands that one
# indicator to the next vendor in the chain and mixes two price sources inside
# a single run.
MAX_LOOK_BACK_DAYS = 275

TRUNCATION_NOTE = (
    "Note: the price history was truncated to the newest sessions; values at the start of "
    "the window may be incomplete."
)

CLAMP_NOTE = (
    "Note: the look-back was clamped from {asked} to {days} days, the longest history one "
    "read can warm up."
)


def _warmup_rows(indicator: str) -> int | None:
    """The rows an indicator needs before its first value means anything.

    stockstats encodes the window in the name where the caller chose one
    (``close_200_sma``), and carries a default for the named ones, which is
    where ``boll_ub`` gets the 20 of its band. An indicator whose default is
    several numbers, the MACD triple among them, has no single window, so it
    gets none rather than a guess.
    """
    for segment in indicator.split("_"):
        if segment.isdigit():
            return int(segment)
    for name in (indicator, indicator.rpartition("_")[0]):
        windows = dft_windows(name) if name else None
        if windows is not None and windows.isdigit():
            return int(windows)
    return None


def _with_warmup_blanked(frame: pd.DataFrame, indicator: str) -> pd.Series:
    """The computed series with the rows that precede its warm-up blanked.

    Every stockstats average fills from the first row it holds
    (``min_periods=1``), so a frame shorter than the window renders a plausible
    number where the honest answer is "not known", and the framework's own
    yfinance path, which loads five years, would answer differently for the
    same symbol and day. Blanking the first ``window - 1`` rows sends the gap
    to the N/A branch instead. Row 0 is blanked whatever the indicator, the
    ones with no derivable window included: it is seeded by a single session,
    so no reported value may rest on it.
    """
    series = frame[indicator].copy()
    window = _warmup_rows(indicator)
    series.iloc[: max(window - 1, 1) if window else 1] = float("nan")
    return series


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    if indicator not in SUPPORTED_INDICATORS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Choose from: {', '.join(SUPPORTED_INDICATORS)}"
        )
    notes: list[str] = []
    if look_back_days > MAX_LOOK_BACK_DAYS:
        notes.append(CLAMP_NOTE.format(asked=look_back_days, days=MAX_LOOK_BACK_DAYS))
        look_back_days = MAX_LOOK_BACK_DAYS
    end = date.fromisoformat(curr_date)
    start = end - timedelta(days=look_back_days + WARMUP_CALENDAR_DAYS)
    text = get_stock_data(symbol, start.isoformat(), end.isoformat())
    frame = wrap(_frame_from_csv(text))
    series = _with_warmup_blanked(frame, indicator)
    by_day = {index.date(): value for index, value in series.items()}
    lines = []
    day = end
    first = end - timedelta(days=look_back_days)
    while day >= first:
        # A day the market never traded and a day whose indicator could not be
        # computed are different facts, and the framework says so in different
        # words. Collapsing them would read as a market closure that never was.
        if day not in by_day:
            lines.append(f"{day.isoformat()}: N/A: Not a trading day (weekend or holiday)")
        elif pd.isna(by_day[day]):
            lines.append(f"{day.isoformat()}: N/A")
        else:
            # `.10g` rather than `.6g`: the framework's own path renders
            # `str(value)`, and an exponent where it prints digits reads as a
            # different number to an agent comparing the two.
            lines.append(f"{day.isoformat()}: {float(by_day[day]):.10g}")
        day -= timedelta(days=1)
    header = f"## {indicator} values from {first.isoformat()} to {curr_date}:"
    description = SUPPORTED_INDICATORS[indicator]
    if _parse_csv_header(text).get("Truncated", "").lower() == "true":
        notes.append(TRUNCATION_NOTE)
    if notes:
        description = "\n".join([*notes, description])
    return f"{header}\n\n" + "\n".join(lines) + f"\n\n{description}"


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


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    data = _call(ticker, lambda c: c.get_fundamentals(ticker, curr_date))
    point_in_time = data.get("point_in_time") or {}
    # The newest period the analysis date could have seen, not whatever the
    # list happened to put first: the order is the server's, so reading
    # position 0 as "latest" would date the metrics to an older filing without
    # saying so, and taking the maximum unbounded would date them to a filing
    # published after the analysis date, which is look-ahead bias in a
    # backtest and deterministic rather than incidental.
    periods = data.get("key_metrics") or []
    if curr_date:
        periods = [row for row in periods if str(row.get("date") or "")[:10] <= curr_date]
    latest = max(periods, key=lambda row: row.get("date") or "") if periods else {}
    blocks = {
        "profile": data.get("profile") or {},
        "ratios_ttm": data.get("ratios_ttm") or {},
        "key_metrics": latest,
    }
    lines = [
        f"# Company Fundamentals for {ticker}",
        f"# Data retrieved on: {_retrieved_on()}",
        "",
    ]
    if latest.get("date"):
        lines.append(f"Key metrics as of: {latest['date']}")
    rendered = 0
    for label, block, field in _FUNDAMENTAL_LABELS:
        value = blocks[block].get(field)
        if value is None:
            continue
        note = ""
        # Say so where the number is today's, so the agent never reads a
        # current value as the value on the analysis date. The server's own
        # flag decides, block by block: the undated ones (`profile`,
        # `ratios_ttm`) are the usual case, but a `key_metrics` the server did
        # not bound is the same claim about a different block.
        if curr_date and not point_in_time.get(block, False):
            note = f" (current values, not as of {curr_date})"
        lines.append(f"{label}: {value}{note}")
        rendered += 1
    if not rendered:
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
        out.append(f"### {item['title']} (source: {item.get('publisher') or 'Unknown'})")
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
