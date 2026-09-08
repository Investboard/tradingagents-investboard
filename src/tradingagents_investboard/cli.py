"""tradingagents-investboard: connect, analyze, status, replay."""

from __future__ import annotations

from datetime import datetime, timezone

import typer

app = typer.Typer(
    help="Connect a locally run TradingAgents analysis to Investboard.", no_args_is_help=True
)


def _today() -> str:
    """Today in the machine's own timezone, as YYYY-MM-DD."""
    return datetime.now(timezone.utc).astimezone().date().isoformat()


@app.command()
def connect() -> None:
    """Connect this machine to your Investboard account (opens the browser once)."""
    from .auth import connect as do_connect

    do_connect()
    typer.echo("Connected. Investboard tokens are stored under ~/.tradingagents/investboard.")


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Exchange-suffixed ticker, e.g. SAP.DE or AAPL"),
    analysis_date: str | None = typer.Option(None, "--date", help="YYYY-MM-DD"),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    checkpoint: bool = typer.Option(
        False, "--checkpoint", help="Enable LangGraph checkpoint resume"
    ),
) -> None:
    """Run TradingAgents and post the result to Investboard."""
    from tradingagents.default_config import DEFAULT_CONFIG

    from .graph import InvestboardTradingAgentsGraph

    # Resolved per invocation. An option default is evaluated once, when the
    # module is imported, so a long-lived process would keep the date it
    # started with.
    analysis_date = analysis_date or _today()
    config = DEFAULT_CONFIG.copy()
    config["checkpoint_enabled"] = checkpoint
    graph = InvestboardTradingAgentsGraph(debug=False, config=config)
    final_state, signal = graph.propagate(ticker.upper(), analysis_date, asset_type=asset_type)
    typer.echo(f"Agent rating: {signal}")
    outcome = final_state.get("investboard") or {}
    stored = outcome.get("stored")
    if stored:
        check = stored.get("check")
        skipped = stored.get("check_skipped_reason")
        verdict = check["status"] if check else f"none ({skipped})"
        typer.echo(f"Stored in Investboard as {stored['id']}. Mandate check: {verdict}")
    else:
        typer.echo(
            f"Investboard post failed ({outcome.get('error')}). "
            f"Saved to {outcome.get('outbox')}; run: tradingagents-investboard replay"
        )
        raise typer.Exit(code=1)


@app.command()
def status(ticker: str = typer.Argument(..., help="Exchange-suffixed ticker")) -> None:
    """List the runs Investboard holds for one instrument."""
    from .auth import access_token, base_url
    from .client import InvestboardClient

    client = InvestboardClient(base_url(), access_token())
    try:
        data = client.list_runs(ticker.upper())
    finally:
        client.close()
    for item in data.get("items", []):
        check = item.get("check")
        verdict = check["status"] if check else "none"
        typer.echo(f"{item['as_of_date']}  {item['decision']['rating']:<12} mandate: {verdict}")
    if not data.get("items"):
        typer.echo("No runs stored yet.")


@app.command()
def replay() -> None:
    """Post runs that failed to reach Investboard."""
    from .graph import replay_outbox

    sent = replay_outbox()
    typer.echo(f"Replayed {len(sent)} run(s)." if sent else "Outbox is empty.")


if __name__ == "__main__":
    app()
