from __future__ import annotations

import httpx
import pytest

from tradingagents_investboard import context
from tradingagents_investboard.client import InvestboardApiError, InvestboardClient

POLICY = {
    "markdown": "# investing.md\n\nCore satellite.",
    "schema_version": "0.4",
    "composed_at": "2026-09-09T10:00:00.000Z",
}
POSITION = {
    "subject": {
        "ticker": "SAP.DE",
        "name": "SAP SE",
        "exchange": "XETRA",
        "isin": "DE0007164600",
        "asset_type": "stock",
        "subject_key": "isin:DE0007164600",
    },
    "held": True,
    "portfolios": [
        {
            "portfolio_name": "Depot A",
            "quantity": 100,
            "weight_pct_of_portfolio": 33.33,
            "cost_basis_native_cents": 1205000,
            "cost_basis_currency": "EUR",
            "market_value_base_cents": 1500000,
            "base_currency": "EUR",
            "first_acquired_at": "2024-03-01T00:00:00.000Z",
        }
    ],
    "household_weight_pct": 20.0,
    "asset_class": "stocks",
    "band": {"min_pct": 40.0, "max_pct": 70.0, "current_pct": 50.0},
    "as_of": "2026-09-09T08:00:00.000Z",
}


def client_for(handler) -> InvestboardClient:
    return InvestboardClient("https://app.example", "tok", transport=httpx.MockTransport(handler))


def refusal(status: int, reason: str | None, hint: str | None = None) -> httpx.Response:
    """A refusal whose ``message`` is not its ``reason``, so a test can tell them apart."""
    details: dict[str, str] = {}
    if reason:
        details["reason"] = reason
    if hint:
        details["hint"] = hint
    error: dict[str, object] = {"code": "NOT_FOUND", "message": "the server refused the read"}
    if details:
        error["details"] = details
    return httpx.Response(status, json={"data": None, "error": error, "meta": {}})


def test_the_policy_block_carries_the_document_verbatim():
    block = context.render_policy_block(POLICY)
    assert block.startswith(
        "Investment policy of the owner (investing.md schema 0.4, "
        "composed 2026-09-09T10:00:00.000Z)."
    )
    assert "Core satellite." in block
    assert "The policy binds the trader" in block


def test_the_position_block_describes_the_holding_in_plain_sentences():
    block = context.render_position_block(POSITION)
    assert "The owner holds SAP.DE" in block
    assert (
        "Depot A: 100 units, 33.33% of that portfolio, market value 15000.00 EUR, "
        "cost basis 12050.00 EUR, first acquired 2024-03-01." in block
    )
    assert "20.0% of the household" in block
    assert (
        "Asset class stocks: 50.0% of the household as of 2026-09-09T08:00:00.000Z, "
        "mandate band 40.0% to 70.0%." in block
    )
    assert "as of 2026-09-09T08:00:00.000Z" in block


def test_an_empty_position_says_so():
    empty = {
        **POSITION,
        "held": False,
        "portfolios": [],
        "household_weight_pct": 0,
        "band": None,
        "asset_class": None,
    }
    block = context.render_position_block(empty)
    assert "The owner does not hold SAP.DE." in block
    assert "mandate band" not in block


def test_a_money_field_the_server_left_out_is_named_not_invented():
    """Nothing is rendered as 0.00 or dropped: the absence is said in words."""
    row = {**POSITION["portfolios"][0], "cost_basis_native_cents": None, "first_acquired_at": None}
    block = context.render_position_block({**POSITION, "portfolios": [row]})

    assert "cost basis not on file." in block
    assert "no cost basis on file" not in block
    assert "first acquired" not in block


def test_a_holding_the_server_could_not_value_says_so():
    """`held` with no row is a state the server states deliberately.

    `buildPositionBlock` reports a holding whose value could not be measured as
    held with no row, rather than as not held or as worth nothing, and its
    household weight is then 0 because the weights count priced holdings only.
    Rendering the header for that block leaves a colon with nothing under it
    and asserts a 0% weight the owner does not have.
    """
    block = context.render_position_block(
        {**POSITION, "portfolios": [], "household_weight_pct": 0.0}
    )

    assert "The owner holds SAP.DE, but no valued position" in block
    assert not block.splitlines()[0].endswith(":")
    # The header form, weight and dangling colon and all, is what must not run.
    assert "The owner holds SAP.DE (" not in block
    assert "(0.0% of the household" not in block


def test_the_freshness_claim_is_the_servers_date_never_the_word_today():
    """The block quotes the server's `as_of`; it never dates anything itself."""
    held = context.render_position_block(POSITION)
    not_held = context.render_position_block({**POSITION, "held": False, "portfolios": []})

    assert "today" not in held
    assert "today" not in not_held
    # The not-held path carries no other date, so the asset-class line is where
    # the block says when it was measured.
    assert "as of 2026-09-09T08:00:00.000Z" in not_held


def test_the_blocks_are_fetched_and_joined_after_the_framework_context():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/context/policy"):
            return httpx.Response(200, json={"data": POLICY, "error": None, "meta": {}})
        return httpx.Response(200, json={"data": POSITION, "error": None, "meta": {}})

    text = context.instrument_context_blocks(client_for(handler), "SAP.DE")
    assert text.startswith("\n\nInvestment policy of the owner")
    assert "\n\nThe owner holds SAP.DE" in text


def test_no_policy_is_stated_and_out_of_scope_is_raised_with_the_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/context/policy"):
            return refusal(404, "no_policy")
        return httpx.Response(200, json={"data": POSITION, "error": None, "meta": {}})

    text = context.instrument_context_blocks(client_for(handler), "SAP.DE")
    assert "No investment policy is on file" in text
    assert "The owner holds SAP.DE" in text

    def refused(request: httpx.Request) -> httpx.Response:
        return refusal(403, "subject_out_of_scope", hint="POST /api/v1/agent/subjects")

    with pytest.raises(InvestboardApiError) as raised:
        context.instrument_context_blocks(client_for(refused), "SAP.DE")
    assert "POST /api/v1/agent/subjects" in raised.value.message


@pytest.mark.parametrize("reason", [None, "run_not_found"], ids=["no reason", "another reason"])
def test_a_404_that_does_not_say_no_policy_is_an_ordinary_refusal(reason):
    """Only the reason means "no mandate on file"; a bare 404 means nothing of the sort.

    The runs read answers a plain 404 for an id it does not hold, so a status
    test here would report an unrelated refusal as an empty mandate and run the
    whole analysis with no policy in anyone's context.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # Only the policy read refuses: a position read that refused as well
        # would raise on its own and the assertion below would pass whatever
        # the policy branch did with the 404.
        if request.url.path.endswith("/context/policy"):
            return refusal(404, reason)
        return httpx.Response(200, json={"data": POSITION, "error": None, "meta": {}})

    with pytest.raises(InvestboardApiError) as raised:
        context.instrument_context_blocks(client_for(handler), "SAP.DE")
    assert raised.value.status == 404


@pytest.mark.parametrize(
    ("failing", "named"),
    [("/context/policy", "investment policy"), ("/context/position", "position")],
    ids=["the policy read", "the position read"],
)
def test_a_read_that_never_completed_names_investboard_the_read_and_the_remedy(failing, named):
    """A transport failure is the one failure with no server message of its own.

    Every refusal arrives with the server's sentence and, for an out-of-scope
    subject, its hint. A dropped connection raises `httpx.ConnectError`, which
    is not an `InvestboardApiError`, so without this it reaches the CLI as
    `Error: All connection attempts failed` and names neither Investboard, nor
    which read failed, nor what the user should do.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if failing in request.url.path:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        body = POLICY if request.url.path.endswith("/context/policy") else POSITION
        return httpx.Response(200, json={"data": body, "error": None, "meta": {}})

    with pytest.raises(RuntimeError) as raised:
        context.instrument_context_blocks(client_for(handler), "SAP.DE")

    message = str(raised.value)
    assert "Investboard" in message
    assert named in message
    assert "run the analysis again" in message
    # The original failure stays attached: the sentence says what to do, the
    # cause says what httpx saw.
    assert isinstance(raised.value.__cause__, httpx.TransportError)
