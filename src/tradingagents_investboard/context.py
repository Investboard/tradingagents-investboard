"""The two blocks Investboard appends to the framework's instrument context.

The framework threads ``instrument_context`` to every agent through
``get_instrument_context_from_state``; ``InvestboardTradingAgentsGraph``
overrides ``resolve_instrument_context`` to append these, so no prompt file
is touched. The policy block is the owner's investing.md document, unchanged;
the position block is the owner's own holding of the instrument, in plain
sentences the agents can quote.

The two say different kinds of thing, and the wording keeps them apart: the
policy is what the owner declared, the position is what the books measure on a
stated date. Neither is advice, and nothing here is computed. Every number is
the server's, rendered in the currency the server named it in; a field the
server left out is said in words rather than filled with a zero.
"""

from __future__ import annotations

from typing import Any

from .client import InvestboardApiError, InvestboardClient

NO_POLICY_NOTE = (
    "No investment policy is on file for the owner; no mandate check will run on this run."
)


def _money(cents: int | None, currency: str | None, missing: str) -> str:
    """A cents integer as an amount in the currency the server named for it.

    The currency travels with the amount because the two money fields are not
    in the same one: market value is in the household's base currency and cost
    basis in the currency the lot was bought in. Rendering either without its
    own label would invite an agent to net them.
    """
    if cents is None or currency is None:
        return missing
    return f"{cents / 100:.2f} {currency}"


def render_policy_block(document: dict[str, Any]) -> str:
    """The owner's investing.md, verbatim, under one sentence of framing."""
    return (
        f"Investment policy of the owner (investing.md schema {document['schema_version']}, "
        f"composed {document['composed_at']}). The policy binds the trader and the portfolio "
        f"manager: a proposal outside its bands is reported as such, never softened.\n\n"
        f"{document['markdown'].rstrip()}"
    )


def render_position_block(block: dict[str, Any]) -> str:
    """What the owner actually holds, as measured on the server's ``as_of`` date."""
    ticker = block["subject"]["ticker"]
    lines: list[str] = []
    if not block.get("held"):
        lines.append(f"The owner does not hold {ticker}. Any proposal is a new position.")
    else:
        lines.append(
            f"The owner holds {ticker} ({block['household_weight_pct']}% of the household, "
            f"as of {block['as_of']}):"
        )
        for row in block.get("portfolios", []):
            acquired = (
                f", first acquired {row['first_acquired_at'][:10]}"
                if row.get("first_acquired_at")
                else ""
            )
            market_value = _money(
                row.get("market_value_base_cents"),
                row.get("base_currency"),
                "no market value on file",
            )
            cost_basis = _money(
                row.get("cost_basis_native_cents"),
                row.get("cost_basis_currency"),
                "no cost basis on file",
            )
            lines.append(
                f"- {row['portfolio_name']}: {row['quantity']} units, "
                f"{row['weight_pct_of_portfolio']}% of that portfolio, "
                f"market value {market_value}, cost basis {cost_basis}{acquired}."
            )
    band = block.get("band")
    if band and block.get("asset_class"):
        lines.append(
            f"Asset class {block['asset_class']}: {band['current_pct']}% of the household today, "
            f"mandate band {band['min_pct']}% to {band['max_pct']}%."
        )
    return "\n".join(lines)


def instrument_context_blocks(client: InvestboardClient, ticker: str) -> str:
    """Both blocks, each opened with a blank line so they sit after the framework's paragraph.

    No policy on file is stated, not raised: an owner without a mandate is a
    normal state, the run still has value, and the policy check later records
    ``no_policy``. That case is the server's ``no_policy`` reason and nothing
    else. Keying on the 404 status instead would be wrong in both directions:
    the API answers a plain 404 for other things, so an unrelated refusal would
    reach the agents as "this owner has no mandate" and the whole analysis
    would run unbound by one that exists.

    An instrument outside the data scope is raised with the server's hint,
    because every data read that follows would refuse the same way; the CLI
    prints the next step.
    """
    try:
        policy = render_policy_block(client.get_policy())
    except InvestboardApiError as error:
        if error.reason != "no_policy":
            raise
        policy = NO_POLICY_NOTE
    position = render_position_block(client.get_position(ticker))
    return f"\n\n{policy}\n\n{position}"
