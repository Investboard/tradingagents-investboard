# tradingagents-investboard

Connect a locally run [TradingAgents](https://github.com/TauricResearch/TradingAgents) analysis to
your [Investboard](https://app.investboard.de) account.

TradingAgents is a multi-agent research framework that you run on your own machine, on your own LLM
keys. This package wraps it so that when a run finishes, the completed analysis is posted to your
Investboard account, where it is stored next to your portfolio and checked against your mandate.

The analysis itself is unchanged. This package adds one HTTP post at the end of a run.

## Requirements

- Python 3.10 or newer
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

This opens your browser once, you sign in to Investboard and approve access. Tokens are written to
`~/.tradingagents/investboard/tokens.json` with owner-only permissions. The access token is short
lived and is refreshed automatically on later runs, so you only connect once per machine.

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
