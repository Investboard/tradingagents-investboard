"""The two blocks Investboard appends to the framework's instrument context.

The framework threads ``instrument_context`` to every agent through
``get_instrument_context_from_state``; ``InvestboardTradingAgentsGraph``
overrides ``resolve_instrument_context`` to append these, so no prompt file
is touched. The policy block is the owner's investing.md document, unchanged;
the position block is the owner's own holding of the instrument, in plain
sentences the agents can quote.

The two say different kinds of thing, and the wording keeps them apart: the
policy is what the owner declared, the position is what the books measure on a
stated date. Neither is advice, and no figure is derived here: the only
arithmetic is the division that renders the server's integer cents as an
amount. Every number is the server's, in the currency the server named it in,
and every date is the server's too.

A money amount the server omitted is said in words rather than filled with a
zero, and so is a weight it could not measure. Of the fields this module
renders, the position contract makes the cost basis, the first-acquired date,
the asset class, its band and the household weight nullable. That last one is
null where no holding in the household carried a countable value: with no
denominator, zero per cent would read as a negligible position rather than as
a book that could not be valued, so it is said in words too. A row's quantity,
its weight of its own portfolio and its market value are required, and are
rendered as they arrive. Where the server sends its coverage counts, the
weight they belong to is quoted with them, because a percentage nobody can see
the denominator of cannot be read.
"""

from __future__ import annotations

from typing import Any

import httpx

from .client import InvestboardApiError, InvestboardClient

NO_POLICY_NOTE = (
    "No investment policy is on file for the owner; no mandate check will run on this run."
)
NOT_ON_FILE = "not on file"
UNMEASURED_HOUSEHOLD_WEIGHT = (
    "The household weight could not be measured: no holding in the household carried a "
    "countable value"
)


def _money(cents: int | None, currency: str | None) -> str:
    """A cents integer as an amount in the currency the server named for it.

    The currency travels with the amount because the two money fields are not
    in the same one: market value is in the household's base currency and cost
    basis in the currency the lot was bought in. Rendering either without its
    own label would invite an agent to net them.

    An absence is the same two words for both, so the sentence reads "market
    value not on file" rather than saying the field's own name back to itself.
    """
    if cents is None or currency is None:
        return NOT_ON_FILE
    return f"{cents / 100:.2f} {currency}"


def _coverage(counts: dict[str, Any] | None) -> str:
    """The book a weight was measured over, as the server's own two counts.

    The server publishes these precisely so a weight is never read without its
    denominator: "100% of that portfolio" is one lone holding or one priced
    holding beside an unvalued one, and the percentage alone cannot tell the
    two apart.

    The block carries the household's counts and each row its own portfolio's,
    under one field name at two scopes, so each call site passes the counts of
    the book its own weight names. Both are optional on the wire, and an absent
    pair renders nothing rather than a denominator this module made up.
    """
    if not counts:
        return ""
    total = counts["total_count"]
    holdings = "holding" if total == 1 else "holdings"
    return f" ({counts['priced_count']} of {total} {holdings} priced)"


def render_policy_block(document: dict[str, Any]) -> str:
    """The owner's investing.md under one sentence of framing.

    The document's own text is unchanged but for the trailing whitespace this
    strips, so the block ends on the last line the owner wrote.
    """
    return (
        f"Investment policy of the owner (investing.md schema {document['schema_version']}, "
        f"composed {document['composed_at']}). The policy binds the trader and the portfolio "
        f"manager: a proposal outside its bands is reported as such, never softened.\n\n"
        f"{document['markdown'].rstrip()}"
    )


def render_position_block(block: dict[str, Any]) -> str:
    """What the owner actually holds, as measured on the server's ``as_of`` date.

    The block asserts no freshness of its own: the line that reports the asset
    class against its band quotes the server's ``as_of`` rather than saying
    "today", which was the one claim this module used to make for itself.
    """
    ticker = block["subject"]["ticker"]
    weight = block.get("household_weight_pct")
    household = _coverage(block.get("coverage"))
    lines: list[str] = []
    if not block.get("held"):
        lines.append(f"The owner does not hold {ticker}. Any proposal is a new position.")
    elif not block.get("portfolios"):
        # The server reports a holding whose value it could not measure as held
        # with no row, rather than as not held or as worth nothing, and the
        # household weight counts priced holdings only, so it arrives as 0, or
        # as null where nothing in the household was priced at all. Rendering
        # the ordinary header here would state that 0 as the owner's weight and
        # then leave the colon with nothing under it.
        lines.append(
            f"The owner holds {ticker}, but no valued position for it could be measured, "
            f"so this block states no quantity, weight or amount for it."
        )
    else:
        # A null weight is the server declining to answer an absent denominator
        # with a zero, and interpolated it prints `None%`. The header then says
        # only what it can, and the sentence below says what it could not.
        measured = (
            f"as of {block['as_of']}"
            if weight is None
            else f"{weight}% of the household{household}, as of {block['as_of']}"
        )
        lines.append(f"The owner holds {ticker} ({measured}):")
        for row in block["portfolios"]:
            acquired = (
                f", first acquired {row['first_acquired_at'][:10]}"
                if row.get("first_acquired_at")
                else ""
            )
            market_value = _money(row.get("market_value_base_cents"), row.get("base_currency"))
            cost_basis = _money(row.get("cost_basis_native_cents"), row.get("cost_basis_currency"))
            lines.append(
                f"- {row['portfolio_name']}: {row['quantity']} units, "
                f"{row['weight_pct_of_portfolio']}% of that portfolio"
                f"{_coverage(row.get('coverage'))}, "
                f"market value {market_value}, cost basis {cost_basis}{acquired}."
            )
    if weight is None:
        lines.append(f"{UNMEASURED_HOUSEHOLD_WEIGHT}{household}.")
    band = block.get("band")
    if band and block.get("asset_class"):
        lines.append(
            f"Asset class {block['asset_class']}: {band['current_pct']}% of the household "
            f"as of {block['as_of']}, mandate band {band['min_pct']}% to {band['max_pct']}%."
        )
    return "\n".join(lines)


def _transport_failure(read: str, error: httpx.TransportError) -> RuntimeError:
    """A read that never completed, said as a sentence rather than httpx's phrase.

    Every refusal on this path arrives carrying the server's own message, and
    an out-of-scope subject carries its hint as well, so a dropped connection
    is the one abort with nothing in it for the reader. Left alone it reaches
    the CLI as "Error: All connection attempts failed", which names neither
    Investboard, nor which of the two reads failed, nor what to do about it.
    """
    return RuntimeError(
        f"Investboard: the {read} read did not complete: {error}. "
        f"No refusal came back, so this is the connection rather than the account: "
        f"wait for it to return and run the analysis again."
    )


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

    A transport failure is named here rather than by routing these two reads
    through ``_session._call``. That seam exists to translate a refusal into
    the framework's vendor taxonomy so the chain can serve the call from
    another provider, and it would turn the deliberate 403 abort above into a
    ``VendorNotConfiguredError`` the chain treats as one vendor being
    misconfigured. These two reads have no other provider, so they translate
    the one error the server never speaks for, and nothing else.
    """
    try:
        policy = render_policy_block(client.get_policy())
    except InvestboardApiError as error:
        if error.reason != "no_policy":
            raise
        policy = NO_POLICY_NOTE
    except httpx.TransportError as error:
        raise _transport_failure("investment policy", error) from error
    try:
        position = render_position_block(client.get_position(ticker))
    except httpx.TransportError as error:
        raise _transport_failure("position", error) from error
    return f"\n\n{policy}\n\n{position}"
