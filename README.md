# tradingagents-investboard

Connect a locally run [TradingAgents](https://github.com/TauricResearch/TradingAgents) analysis to
your [Investboard](https://app.investboard.de) account.

TradingAgents is a multi-agent research framework that you run on your own machine, on your own LLM
keys. This package wraps it so that when a run finishes, the completed analysis is posted to your
Investboard account, where it is stored next to your portfolio and checked against your mandate.

The analysis itself is unchanged. This package adds one HTTP post at the end of a run.

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

## Troubleshooting

**`Error: Not connected. Run: tradingagents-investboard connect`**

The machine has no usable token: it was never connected, the token file was removed, or the stored
refresh token no longer works (you revoked the connection, or the account changed). Run `connect`
again. `analyze` asks for the token before it starts the analysis, so this never costs you a run.

**`Error: Investboard could not be reached while refreshing the connection`**

The refresh got no answer: no network, or Investboard was briefly unavailable. Your connection is
intact and `connect` is not the remedy. Run the command again in a moment.

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
the post. The result of the post is available under `final_state["investboard"]`.

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

Only the finished run described above is posted, and only to your own Investboard account.

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
