# tradingagents-investboard

Connect a locally run [TradingAgents](https://github.com/TauricResearch/TradingAgents) analysis to
your [Investboard](https://app.investboard.de) account.

TradingAgents is a multi-agent research framework that you run on your own machine, on your own LLM
keys. This package wraps it so that when a run finishes, the completed analysis is posted to your
Investboard account, where it is stored next to your portfolio and checked against your mandate.

The analysis itself stays the framework's. This package adds three things to it: your own data
behind the framework's core data reads, your policy and your position in every agent's context, and
one HTTP post at the end of the run.

## Requirements

- Python 3.10 or newer. Check with `python3 --version`; on macOS, `brew install python@3.12` if
  the system Python is older
- Your own LLM provider key, the same one TradingAgents already needs
- An Investboard account

## Install

```bash
pip install git+https://github.com/Investboard/tradingagents-investboard.git
```

The install pulls TradingAgents and LangChain, so it takes a few minutes.

## Connect

```bash
tradingagents-investboard connect
```

This prints the authorization URL and opens it in your browser, you sign in to Investboard and
approve access. On a machine with no browser, copy the printed URL. Tokens are written to
`~/.tradingagents/investboard/tokens.json` with owner-only permissions.

You connect once per machine. An access token lives a day; when it has run out, the next command
spends the stored refresh token at Investboard's token endpoint and writes the new pair back to the
same file. The refresh token is long-lived, so no later run needs a browser.

## Analyse

```bash
tradingagents-investboard analyze SAP.DE
tradingagents-investboard analyze AAPL --date 2026-09-08
tradingagents-investboard analyze BTC-USD --asset-type crypto
tradingagents-investboard analyze NVDA --register           # research one you do not hold
tradingagents-investboard analyze SAP.DE --vendor default   # the framework's own data vendors
```

The run prints the agent rating and then confirms what Investboard stored, including the mandate
check verdict.

Two more commands:

```bash
tradingagents-investboard status SAP.DE   # the runs Investboard holds for one instrument
tradingagents-investboard replay          # post runs that could not reach Investboard
```

If a post fails, the run is not lost. The payload is written to
`~/.tradingagents/investboard/outbox/` and `replay` sends it later. Posts are idempotent, so
replaying a run that already arrived does not duplicate it.

## Data

By default (`--vendor investboard`) the framework's core data categories are served by your
Investboard account instead of by its own vendors:

- daily price history (OHLCV) for the window an agent asks for
- the technical indicators, computed on your machine with `stockstats` from that price history,
  which is why there is no separate indicator read
- company fundamentals: profile, trailing-twelve-month ratios and key metrics
- income statement, balance sheet and cash flow, annual or quarterly
- company news over a date window
- insider transactions over a date window

The reads are scoped to instruments you already hold, watch, or registered. Anything else is
refused, and the refusal names registration as the remedy. To research an instrument outside that
scope, register it first:

```bash
tradingagents-investboard analyze NVDA --register
```

Registration is idempotent: a second one reports the instrument as already registered. Investboard
allows twenty registrations and 2,000 data reads per UTC day.

Nothing dated after the analysis date is served where the read carries a date: the price history,
the news window, the dated fundamentals rows and the statement periods all stop there, so a run
dated in the past does not read a later filing, a later session or later news. Two exceptions are
named rather than hidden. The company profile and the trailing-twelve-month ratios carry no date
of their own, so they are current values, and each line says so where it is served for a past
date. Insider transactions are the one read the framework asks for with no date at all, so they
cover the twelve months up to today rather than up to the analysis date.

What Investboard does not serve, and what serves it instead:

- global news comes from the framework's own vendor, which is why the `news_data` category is the
  chain `investboard,yfinance` rather than `investboard` alone
- macroeconomic data (FRED) and prediction markets (Polymarket) stay on the framework's vendors,
  untouched by this package
- the retail-sentiment blocks (StockTwits, Reddit) are fetched by the framework itself, not through
  a vendor at all
- three framework calls bypass vendor routing and read yfinance directly: its verified market
  snapshot, the instrument identity lookup, and the realised returns its reflection layer reads
  after the fact

`--vendor default` leaves every category on the framework's own vendors, which is the way to
compare a run against the framework's unmodified data. The scope rule still applies to it: the
position block below is read at the start of every run, whichever vendor serves the data.

Either way, every analyst, researcher and manager in the run reads two blocks appended to the
framework's instrument context: your investment policy (`investing.md`, as Investboard composed
it) and your position in the instrument, per portfolio, with quantity, weight, cost basis and
market value, the weight of your whole household, and the mandate band for the asset class where
your mandate sets one. Nothing in those blocks is computed here: the numbers are the ones
Investboard holds, on the date it states. An owner with no policy on file is told so in one line
and the run continues.

## Troubleshooting

**`Error: Not connected. Run: tradingagents-investboard connect`**

The machine has no usable token: it was never connected, the token file was removed, or the stored
refresh token no longer works (you revoked the connection, or the account changed). Run `connect`
again. `analyze` asks for the token before it starts the analysis, so this never costs you a run.

**`Error: Investboard could not be reached while refreshing the connection`**

The refresh got no answer: no network, or Investboard was briefly unavailable. Your connection is
intact and `connect` is not the remedy. Run the command again in a moment.

**An instrument outside your scope (`subject_out_of_scope`)**

The instrument is not one you hold, not on a watchlist and not registered, so Investboard serves
neither data nor a position for it. The position block is read before the first agent runs, so this
costs you no model time. Either add the instrument to a watchlist in Investboard, or run `analyze`
again with `--register`.

**A run finished but the post did not**

The payload is in `~/.tradingagents/investboard/outbox/`, one JSON file per run, and nothing has
been lost. Run `tradingagents-investboard replay`. It reports what it sent, what Investboard
refused, and what is still queued, and it exits 1 if anything did not go through. A network outage
leaves every file in place, so simply run it again when the connection is back.

**A queued entry**

A refusal about your account rather than the run stays in the queue: a lapsed connection, a paused
subscription, a daily cap already reached. `replay` stops at that entry and leaves it, and every
entry behind it, exactly where it is. Nothing to rename: clear the cause, then run `replay` again.

**A rejected entry**

A refusal Investboard will repeat for the run itself is set aside rather than retried: a payload it
cannot parse, a ticker it cannot resolve, a run over the size limit. `replay` renames those to
`<run id>.json.rejected` and moves on, so one bad entry cannot block the queue. Nothing is deleted.
Delete the file when you no longer want the run, or drop the `.rejected` suffix to put it back in
the queue.

## Use it from Python

```python
from tradingagents.default_config import DEFAULT_CONFIG

from tradingagents_investboard.graph import InvestboardTradingAgentsGraph

graph = InvestboardTradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
final_state, signal = graph.propagate("SAP.DE", "2026-09-08")
print(signal, final_state["investboard"])
```

`InvestboardTradingAgentsGraph` is a subclass of `TradingAgentsGraph` with the same behaviour, plus
the post and the two context blocks. The result of the post is available under
`final_state["investboard"]`.

That run reads the framework's own data vendors, because nothing pointed them elsewhere. To read
Investboard data as `analyze` does, import the vendor module, which registers it, and name it in a
copy of the vendor mapping rather than in the framework's own:

```python
# Importing the module is what registers the vendor with the framework.
from tradingagents_investboard import vendor  # noqa: F401

config = DEFAULT_CONFIG.copy()
config["data_vendors"] = dict(DEFAULT_CONFIG["data_vendors"])
config["data_vendors"].update(
    {
        "core_stock_apis": "investboard",
        "technical_indicators": "investboard",
        "fundamental_data": "investboard",
        "news_data": "investboard,yfinance",
    }
)
graph = InvestboardTradingAgentsGraph(debug=False, config=config)
```

## What Investboard receives

One record per completed run:

- the ticker, the analysis date and when the run started and finished
- the final decision: rating, executive summary, investment thesis, price target, time horizon
- the trader proposal, when the run produced one: action, reasoning, entry price, stop loss, sizing
- the four analyst reports: market, sentiment, news, fundamentals
- the bull and bear debate and the research manager's call
- the risk debate and the portfolio manager's decision
- which models you used, and how many debate rounds you allowed

A decision the parser cannot read is sent with the rating `REVIEW` and the raw text as its summary.
It is never coerced into a tradeable rating.

## What never leaves your machine

- your LLM provider keys, which this package never reads and never transmits
- the agent prompts and the framework's internal reasoning traffic
- the market data the framework pulls while it works

Only the finished run described above is posted, and only to your own Investboard account. The
data reads go the other way: they ask Investboard for one instrument by ticker, with the date
window and, for a statement, the period, and the answer comes back to your machine.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVESTBOARD_BASE_URL` | `https://app.investboard.de` | Point the client at another environment |
| `TRADINGAGENTS_INVESTBOARD_HOME` | `~/.tradingagents/investboard` | Where tokens and the outbox live |

## Licence

Apache 2.0, see [LICENSE](LICENSE). TradingAgents is Apache 2.0 and belongs to
[Tauric Research](https://github.com/TauricResearch).

Investboard is a tool for investment strategy and analysis. Nothing it stores or shows is financial
advice, and it does not place orders.
