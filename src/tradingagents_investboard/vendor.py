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
from stockstats import wrap
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.interface import VENDOR_LIST, VENDOR_METHODS

from ._session import _call

try:
    from stockstats import dft_windows as _stockstats_dft_windows
except ImportError:  # pragma: no cover - a 0.6.x the pin permits but that lacks it
    # `dft_windows` is undocumented in 0.6.8 and the dependency is pinned
    # `>=0.6,<1`, so a permitted resolution could not carry it. Imported at
    # module scope it would then fail this whole module, and with it the CLI
    # that imports it for the registration side effect. The fallback below
    # keeps the warm-up floors rather than silently losing them.
    _stockstats_dft_windows = None

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


SESSION_DATE_FORMAT = "%Y-%m-%d"


def _session_days(column: pd.Series) -> pd.Series:
    """The Date column as the calendar days the server wrote, or a refusal.

    The server's Date is a bare ``YYYY-MM-DD`` calendar day by contract (its
    bar schema types it as Zod's ``.date()``, and the CSV writer emits that
    string unchanged), and the whole render rests on it: sessions are keyed by
    ``.date()``. Anything wider re-dates them. Read with ``utc=True``, a
    Frankfurt session stamped ``2026-01-12T00:00:00+01:00`` normalises to
    ``2026-01-11T23:00Z``, whose date is the Sunday: every session shifts back
    a day, the Monday the market traded prints "not a trading day" and the
    Sunday it did not prints a number. That is the exact lie the holiday
    branch exists to prevent.

    So the format is pinned, which also removes the mixed-offset ``ValueError``
    ``utc=True`` was added for: a column of bare days has no offsets to mix. A
    stamped row is refused rather than moved, because moving it correctly
    needs the exchange's own timezone and neither this package nor the payload
    carries one: the date of the stamp's local part is right only where the
    stamp is exchange-local and wrong where it is already UTC, so normalising
    would trade one silent mis-dating for another.
    """
    parsed = pd.to_datetime(column, format=SESSION_DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        offending = column[parsed.isna()].astype(str).tolist()[:3]
        raise ValueError(
            "Investboard: the OHLCV Date column must be a bare YYYY-MM-DD calendar day; "
            f"refusing rather than re-dating {offending}"
        )
    return parsed


def _frame_from_csv(text: str) -> pd.DataFrame:
    body = _split_csv(text)[1]
    frame = pd.read_csv(io.StringIO(body))
    frame["Date"] = _session_days(frame["Date"])
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

# The margins above are thin, so the mask can reach past the warm-up and into
# the reported window: a market printing under about 243 sessions a year, or a
# seven-day instrument whose oldest rows the server dropped. `close_200_sma`
# over a 275-day look-back is the worst of them, and about 74 reported days
# then read N/A where the framework's yfinance path prints numbers. Both
# blanked states print the same "N/A", so without this the reader cannot tell
# "the history behind it is short" from "the value could not be computed".
WARMUP_NOTE = (
    "Note: the warm-up reaches into this window: N/A on its {oldest} means the history there "
    "is shorter than the indicator needs, not that the value could not be computed."
)

# The server pulls its `to` back to the last completed UTC day, so on a default
# run the analysis date itself is served by no row however busily the market is
# trading. Reported under the closure sentence below, that reads as a holiday
# the market never had, so a day past the last row served says what it is. The
# line names the last session served, which is the whole of what is known here:
# on stale or delisted history the market has completed sessions the server did
# not send, and calling that row the last completed one would say something
# about the market that this package cannot see.
UNSERVED_LINE = "N/A: after the last session served ({last})"

# A null cell is not blanked. A missing volume among valid rows is skipped by
# the rolling sums rather than propagated, so the VWMA of that session and the
# thirteen after it is computed from fewer rows and still printed as a number.
# Blanking on any null would over-mask every indicator that never reads the
# missing column, and a per-indicator input map would guess at stockstats'
# internals rather than read them, so the gaps are counted and declared.
INCOMPLETE_NOTE = (
    "Note: the served history has {gaps} with a price or volume cell missing; values within "
    "one window of such a session are computed from fewer rows than the window names."
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _incomplete_sessions(frame: pd.DataFrame) -> int:
    """Served sessions missing any one of the OHLCV cells."""
    columns = [column for column in _OHLCV_COLUMNS if column in frame.columns]
    return int(frame[columns].isna().any(axis=1).sum()) if columns else 0


# stockstats' own defaults for the named indicators this vendor supports, as
# it holds them in 0.6.8. Read only where the `dft_windows` import above found
# nothing, so a resolution without that helper keeps the warm-up floors rather
# than dropping every indicator to the row-0 mask.
_FALLBACK_WINDOWS = {
    "macd": "12,26,9",
    "boll": "20",
    "rsi": "14",
    "atr": "14",
    "vwma": "14",
    "mfi": "14",
}


def _default_windows(name: str) -> str | None:
    if _stockstats_dft_windows is not None:
        return _stockstats_dft_windows(name)
    return _FALLBACK_WINDOWS.get(name)


def _largest_window(windows: str) -> int | None:
    """The longest period in a stockstats default, which may be a comma tuple."""
    numbers = [int(part) for part in windows.split(",") if part.strip().isdigit()]
    return max(numbers) if numbers else None


def _warmup_rows(indicator: str) -> int | None:
    """The rows an indicator needs before its first value means anything.

    stockstats encodes the window in the name where the caller chose one
    (``close_200_sma``), and carries a default for the named ones, which is
    where ``boll_ub`` gets the 20 of its band.

    A default of several numbers is read as its largest rather than skipped.
    ``dft_windows("macd")`` is the string ``"12,26,9"``, and MACD is defined as
    the difference of a 12 and a 26 period EMA: it is not a MACD before 26
    rows, so 26 is a definition and not a guess. ``macds`` and ``macdh`` are
    built from that line and carry no default of their own, so they inherit
    its floor through the trailing-character strip. The signal line's own
    9-period smoothing sits on top of the 26 and is not counted here, which
    makes this a floor rather than the full warm-up.
    """
    for segment in indicator.split("_"):
        if segment.isdigit():
            return int(segment)
    for name in (indicator, indicator.rpartition("_")[0], indicator[:-1]):
        windows = _default_windows(name) if name else None
        largest = _largest_window(windows) if windows else None
        if largest is not None:
            return largest
    return None


def _warmup_span(indicator: str) -> int:
    """The leading rows the warm-up blanks: the window bar its last row.

    At least one whatever the indicator, the ones with no derivable window
    included: row 0 is seeded by a single session, so no reported value may
    rest on it.
    """
    window = _warmup_rows(indicator)
    return max(window - 1, 1) if window else 1


def _with_warmup_blanked(frame: pd.DataFrame, indicator: str) -> pd.Series:
    """The computed series with the rows that precede its warm-up blanked.

    Every stockstats average fills from the first row it holds
    (``min_periods=1``), so a frame shorter than the window renders a plausible
    number where the honest answer is "not known", and the framework's own
    yfinance path, which loads five years, would answer differently for the
    same symbol and day. Blanking the first ``window - 1`` rows sends the gap
    to the N/A branch instead.
    """
    series = frame[indicator].copy()
    series.iloc[: _warmup_span(indicator)] = float("nan")
    return series


def _warmup_reached_note(
    frame: pd.DataFrame, indicator: str, first: date, end: date
) -> str | None:
    """The note for a warm-up that ran past the history and into the window.

    The blanked rows are the oldest the frame holds, so a blanked row dated on
    or after the first reported day is a reported day the mask took.
    """
    masked = [index.date() for index in frame.index[: _warmup_span(indicator)]]
    count = sum(1 for day in masked if first <= day <= end)
    if not count:
        return None
    oldest = "oldest session" if count == 1 else f"{count} oldest sessions"
    return WARMUP_NOTE.format(oldest=oldest)


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
    served = _frame_from_csv(text)
    # Counted before the wrap, which adds the computed columns to the frame.
    incomplete = _incomplete_sessions(served)
    # `get_stock_data` refuses an empty body, so the frame has a last session.
    last_session = served.index[-1].date()
    frame = wrap(served)
    series = _with_warmup_blanked(frame, indicator)
    by_day = {index.date(): value for index, value in series.items()}
    lines = []
    day = end
    first = end - timedelta(days=look_back_days)
    while day >= first:
        # A day the market never traded, a day the server has not served yet
        # and a day whose indicator could not be computed are three different
        # facts, and each is said in its own words. Collapsing any two of them
        # would read as a market closure that never was.
        if day > last_session:
            lines.append(f"{day.isoformat()}: {UNSERVED_LINE.format(last=last_session)}")
        elif day not in by_day:
            lines.append(f"{day.isoformat()}: N/A: Not a trading day (weekend or holiday)")
        elif pd.isna(by_day[day]):
            lines.append(f"{day.isoformat()}: N/A")
        else:
            # `.10g` rather than `.6g`: the framework's own path renders
            # `str(value)`, and an exponent where it prints digits reads as a
            # different number to an agent comparing the two. `.10g` widens
            # the range over which that holds; it does not remove the
            # exponent. It still switches to one below 1e-4, so a sub-penny
            # value renders `1.2345e-05`. The claim is decimal digits for
            # values from 1e-4 up to the tenth significant figure, which is
            # every price and average an equity instrument prints.
            lines.append(f"{day.isoformat()}: {float(by_day[day]):.10g}")
        day -= timedelta(days=1)
    header = f"## {indicator} values from {first.isoformat()} to {curr_date}:"
    description = SUPPORTED_INDICATORS[indicator]
    if _parse_csv_header(text).get("Truncated", "").lower() == "true":
        notes.append(TRUNCATION_NOTE)
    reached = _warmup_reached_note(frame, indicator, first, end)
    if reached:
        notes.append(reached)
    if incomplete:
        gaps = "1 session" if incomplete == 1 else f"{incomplete} sessions"
        notes.append(INCOMPLETE_NOTE.format(gaps=gaps))
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
    # Whether each block is as of the analysis date rather than today's value.
    # `point_in_time` is the server's claim block by block, and its
    # `key_metrics` flag means "these rows were cut at `as_of`", not "these
    # were restated to today": the web side sets the three as one literal, and
    # `key_metrics` is the true one precisely because those rows are cut by
    # `as_of`. The undated blocks are the provider's current values and the
    # note is the whole truth about them.
    #
    # `key_metrics` does not read that flag, because this renderer cuts the
    # rows again against `curr_date` above: a dated key metric is as of that
    # date by construction whichever way the flag reads, and printing the note
    # as well would put "as of 2025-12-31" and "current values, not as of
    # 2026-09-08" beside each other about one number, where one of the two has
    # to be false. An undated row is the one key metric nothing here can
    # bound, there is no "as of" line for it, and the note is then true.
    as_of_bound = {
        "profile": bool(point_in_time.get("profile", False)),
        "ratios_ttm": bool(point_in_time.get("ratios_ttm", False)),
        "key_metrics": bool(latest.get("date")),
    }
    rendered = 0
    for label, block, field in _FUNDAMENTAL_LABELS:
        value = blocks[block].get(field)
        if value is None:
            continue
        note = ""
        if curr_date and not as_of_bound[block]:
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
