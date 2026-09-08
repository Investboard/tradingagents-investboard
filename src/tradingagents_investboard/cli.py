"""tradingagents-investboard: connect, analyze, status, replay."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any, TypeVar

import typer

app = typer.Typer(
    help="Connect a locally run TradingAgents analysis to Investboard.", no_args_is_help=True
)

# Exchange-suffixed tickers (SAP.DE), indices (^GDAXI), pairs (BTC-USD) and the
# odd vendor form. Anything else is a typo, and a typo should not cost an hour
# of model time before the server rejects it. At least one alphanumeric
# character, since "..." and "---" name no instrument, and a ceiling no real
# symbol comes near.
TICKER_RE = re.compile(r"(?=.{1,64}\Z)[A-Za-z0-9.^@_/-]*[A-Za-z0-9][A-Za-z0-9.^@_/-]*")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

F = TypeVar("F", bound=Callable[..., Any])


def _reported(command: F) -> F:
    """Report a failure as one line, not as a traceback.

    A stack trace tells the user nothing they can act on. Every command ends
    either in its own message or in this one.
    """

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except Exception as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1) from error

    return wrapper  # type: ignore[return-value]


def _fail(message: str) -> None:
    """End on stderr, the way `_reported` ends on an exception.

    A failure the command saw coming reads exactly like one it did not: same
    prefix, same stream. Stdout stays for what the command was asked to report.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _today() -> str:
    """Today in the machine's own timezone, as YYYY-MM-DD."""
    return datetime.now(timezone.utc).astimezone().date().isoformat()


def _checked_date(value: str) -> str:
    if not DATE_RE.fullmatch(value):
        _fail(f"--date must be YYYY-MM-DD, not {value!r}.")
    try:
        date.fromisoformat(value)
    except ValueError:
        _fail(f"{value!r} is not a real calendar date.")
    return value


def _checked_ticker(value: str) -> str:
    if not TICKER_RE.fullmatch(value):
        _fail(f"{value!r} is not a usable ticker.")
    return value.upper()


@app.command()
@_reported
def connect() -> None:
    """Connect this machine to your Investboard account (opens the browser once)."""
    from .auth import connect as do_connect

    do_connect()
    typer.echo("Connected. Investboard tokens are stored under ~/.tradingagents/investboard.")


@app.command()
@_reported
def analyze(
    ticker: str = typer.Argument(..., help="Exchange-suffixed ticker, e.g. SAP.DE or AAPL"),
    analysis_date: str | None = typer.Option(None, "--date", help="YYYY-MM-DD"),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    checkpoint: bool = typer.Option(
        False, "--checkpoint", help="Enable LangGraph checkpoint resume"
    ),
) -> None:
    """Run TradingAgents and post the result to Investboard."""
    from .auth import access_token

    # Resolved per invocation. An option default is evaluated once, when the
    # module is imported, so a long-lived process would keep the date it
    # started with.
    analysis_date = _checked_date(analysis_date or _today())
    ticker = _checked_ticker(ticker)
    # Asked for before the first token is spent: an expired connection found
    # after the analysis would cost the whole run.
    access_token()

    from tradingagents.default_config import DEFAULT_CONFIG

    from .graph import InvestboardTradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["checkpoint_enabled"] = checkpoint
    graph = InvestboardTradingAgentsGraph(debug=False, config=config)
    final_state, signal = graph.propagate(ticker, analysis_date, asset_type=asset_type)
    typer.echo(f"Agent rating: {signal}")
    outcome = final_state.get("investboard") or {}
    stored = outcome.get("stored")
    if stored:
        check = stored.get("check")
        skipped = stored.get("check_skipped_reason")
        verdict = check["status"] if check else f"none ({skipped})"
        typer.echo(f"Stored in Investboard as {stored['id']}. Mandate check: {verdict}")
    else:
        _fail(
            f"Investboard post failed ({outcome.get('error')}). "
            f"Saved to {outcome.get('outbox')}; run: tradingagents-investboard replay"
        )


@app.command()
@_reported
def status(ticker: str = typer.Argument(..., help="Exchange-suffixed ticker")) -> None:
    """List the runs Investboard holds for one instrument."""
    from .auth import access_token, base_url
    from .client import InvestboardClient

    client = InvestboardClient(base_url(), access_token())
    try:
        data = client.list_runs(_checked_ticker(ticker))
    finally:
        client.close()
    for item in data.get("items", []):
        check = item.get("check")
        verdict = check["status"] if check else "none"
        typer.echo(f"{item['as_of_date']}  {item['decision']['rating']:<12} mandate: {verdict}")
    if not data.get("items"):
        typer.echo("No runs stored yet.")


@app.command()
@_reported
def replay() -> None:
    """Post runs that failed to reach Investboard."""
    from .graph import replay_outbox

    outcome = replay_outbox()
    if not any(outcome.values()):
        typer.echo("Outbox is empty.")
        return
    typer.echo(
        f"Sent {len(outcome['sent'])}, rejected {len(outcome['rejected'])}, "
        f"still queued {len(outcome['failed'])}."
    )
    for run_id in outcome["rejected"]:
        typer.echo(f"Rejected: {run_id} (kept as .rejected, Investboard will not accept it)")
    if outcome["rejected"] or outcome["failed"]:
        _fail(
            "Not every run reached Investboard. A queued run is retried by the next replay; "
            "a rejected one is kept on disk and needs you."
        )


if __name__ == "__main__":
    app()
