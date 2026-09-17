import sys

import pytest
from typer.testing import CliRunner

from tradingagents_investboard import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def framework_vendors(monkeypatch):
    """Every test meets the framework's vendor defaults as the framework left them.

    `analyze` configures a copy, and the test below says so. The claim would be
    vacuous without this: an earlier test in the file runs `analyze` too, so a
    leak from it would already have written the same choice into the module
    default, and the comparison would then hold for the wrong reason.
    """
    from tradingagents.default_config import DEFAULT_CONFIG

    monkeypatch.setitem(DEFAULT_CONFIG, "data_vendors", dict(DEFAULT_CONFIG["data_vendors"]))


@pytest.fixture
def connected(monkeypatch):
    """A token is available, so nothing below fails for want of a connection."""
    monkeypatch.setattr(cli, "_today", lambda: "2026-09-08")
    import tradingagents_investboard.auth as auth

    monkeypatch.setattr(auth, "access_token", lambda *args, **kwargs: "tok")
    return auth


def test_an_impossible_date_is_refused_before_any_model_runs(connected):
    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-02-30"])

    assert result.exit_code == 1
    assert "not a real calendar date" in result.output


def test_a_misshaped_date_is_refused(connected):
    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "08.09.2026"])

    assert result.exit_code == 1
    assert "YYYY-MM-DD" in result.output


@pytest.mark.parametrize(
    "ticker",
    [
        "SAP DE; rm -rf /",
        # Punctuation is allowed inside a symbol, but it never is the symbol.
        "...",
        "___",
        # And no instrument is named by 65 characters.
        "A" * 65,
    ],
)
def test_a_ticker_that_is_not_a_ticker_is_refused(connected, ticker):
    result = runner.invoke(cli.app, ["analyze", ticker])

    assert result.exit_code == 1
    assert "not a usable ticker" in result.output


def test_analyze_stops_at_the_missing_connection_before_the_llm(monkeypatch):
    """An expired connection found after the analysis would cost the whole run."""
    import tradingagents_investboard.auth as auth

    def refuse(*args, **kwargs):
        raise RuntimeError(auth.NOT_CONNECTED)

    monkeypatch.setattr(auth, "access_token", refuse)

    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08"])

    assert result.exit_code == 1
    assert result.output.strip() == f"Error: {auth.NOT_CONNECTED}"


def test_a_failure_is_one_line_not_a_traceback(monkeypatch):
    import tradingagents_investboard.auth as auth

    def explode(*args, **kwargs):
        raise ValueError("the endpoint moved")

    monkeypatch.setattr(auth, "access_token", explode)

    result = runner.invoke(cli.app, ["status", "SAP.DE"])

    assert result.exit_code == 1
    assert result.output.strip() == "Error: the endpoint moved"
    assert "Traceback" not in result.output


def test_replay_reports_the_counts_and_fails_when_anything_was_rejected(monkeypatch):
    """The counts are the report and stay on stdout; the failure is an error and
    goes where every other error goes, so a pipeline sees it."""
    import tradingagents_investboard.graph as graph

    monkeypatch.setattr(
        graph, "replay_outbox", lambda: {"sent": ["a"], "rejected": ["b"], "failed": []}
    )

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 1
    assert "Sent 1, rejected 1, still queued 0." in result.stdout
    assert "Rejected: b" in result.stdout
    assert result.stderr.startswith("Error: Not every run reached Investboard")


def test_a_failed_post_reports_the_outbox_as_an_error(connected, monkeypatch):
    """The run itself succeeded, so the rating is a result and belongs on
    stdout; the post did not, and that is an error like any other."""
    import tradingagents_investboard.graph as graph

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, ticker, analysis_date, asset_type="stock"):
            outcome = {"stored": None, "outbox": "/outbox/run-1.json", "error": "503 down"}
            return {"investboard": outcome}, "Overweight"

    monkeypatch.setattr(graph, "InvestboardTradingAgentsGraph", FakeGraph)

    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08"])

    assert result.exit_code == 1
    assert result.stdout.strip() == "Agent rating: Overweight"
    assert result.stderr.startswith("Error: Investboard post failed (503 down).")
    assert "/outbox/run-1.json" in result.stderr


def test_replay_of_an_empty_outbox_succeeds(monkeypatch):
    import tradingagents_investboard.graph as graph

    monkeypatch.setattr(graph, "replay_outbox", lambda: {"sent": [], "rejected": [], "failed": []})

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 0
    assert "Outbox is empty." in result.output


def test_analyze_points_the_core_categories_at_investboard_without_touching_the_default(
    connected, monkeypatch
):
    """The run configures a copy: a second run in the same process, and every
    other consumer of the framework's default, must be unaffected."""
    from tradingagents.default_config import DEFAULT_CONFIG

    captured: dict = {}

    class FakeGraph:
        def __init__(self, debug=False, config=None):
            captured["config"] = config

        def propagate(self, ticker, trade_date, asset_type="stock"):
            stored = {"id": "run-1", "check": None, "check_skipped_reason": "no_sizing"}
            return {"investboard": {"stored": stored}}, "Hold"

    monkeypatch.setattr("tradingagents_investboard.graph.InvestboardTradingAgentsGraph", FakeGraph)
    before = dict(DEFAULT_CONFIG["data_vendors"])

    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08"])

    assert result.exit_code == 0, result.output
    vendors = captured["config"]["data_vendors"]
    assert vendors["core_stock_apis"] == "investboard"
    assert vendors["technical_indicators"] == "investboard"
    assert vendors["fundamental_data"] == "investboard"
    # get_global_news is not ours, and a category of "investboard" alone would
    # raise for it, so the chain names yfinance behind us.
    assert vendors["news_data"] == "investboard,yfinance"
    assert DEFAULT_CONFIG["data_vendors"] == before
    assert "investboard" in sys.modules.get("tradingagents.dataflows.interface").VENDOR_LIST


def test_analyze_keeps_the_framework_vendors_on_request(connected, monkeypatch):
    captured: dict = {}

    class FakeGraph:
        def __init__(self, debug=False, config=None):
            captured["config"] = config

        def propagate(self, ticker, trade_date, asset_type="stock"):
            stored = {"id": "run-1", "check": None, "check_skipped_reason": None}
            return {"investboard": {"stored": stored}}, "Hold"

    monkeypatch.setattr("tradingagents_investboard.graph.InvestboardTradingAgentsGraph", FakeGraph)

    result = runner.invoke(
        cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08", "--vendor", "default"]
    )

    assert result.exit_code == 0, result.output
    assert captured["config"]["data_vendors"]["core_stock_apis"] != "investboard"


def test_an_unknown_vendor_is_refused_before_the_analysis(connected, monkeypatch):
    """A typo must not run the whole analysis on the framework's own vendors."""

    class FakeGraph:
        def __init__(self, debug=False, config=None):
            raise AssertionError("the graph must not be built for an unknown vendor")

    monkeypatch.setattr("tradingagents_investboard.graph.InvestboardTradingAgentsGraph", FakeGraph)

    result = runner.invoke(
        cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08", "--vendor", "yfinance"]
    )

    assert result.exit_code == 1
    assert "Unknown vendor yfinance" in result.stderr


def test_analyze_registers_the_subject_first_when_asked(connected, monkeypatch):
    registered: list[str] = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def register_subject(self, ticker):
            registered.append(ticker)
            return {"created": True}

        def close(self):
            pass

    class FakeGraph:
        def __init__(self, debug=False, config=None):
            pass

        def propagate(self, ticker, trade_date, asset_type="stock"):
            stored = {"id": "run-1", "check": None, "check_skipped_reason": None}
            return {"investboard": {"stored": stored}}, "Hold"

    monkeypatch.setattr("tradingagents_investboard.client.InvestboardClient", FakeClient)
    monkeypatch.setattr("tradingagents_investboard.graph.InvestboardTradingAgentsGraph", FakeGraph)

    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08", "--register"])

    assert result.exit_code == 0, result.output
    assert registered == ["SAP.DE"]
    assert "Registered SAP.DE" in result.output


def test_a_second_registration_says_so(connected, monkeypatch):
    """The server answers 200 with created: false; the user should not read
    that as a fresh registration against the daily cap."""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def register_subject(self, ticker):
            return {"created": False}

        def close(self):
            pass

    class FakeGraph:
        def __init__(self, debug=False, config=None):
            pass

        def propagate(self, ticker, trade_date, asset_type="stock"):
            stored = {"id": "run-1", "check": None, "check_skipped_reason": None}
            return {"investboard": {"stored": stored}}, "Hold"

    monkeypatch.setattr("tradingagents_investboard.client.InvestboardClient", FakeClient)
    monkeypatch.setattr("tradingagents_investboard.graph.InvestboardTradingAgentsGraph", FakeGraph)

    result = runner.invoke(cli.app, ["analyze", "SAP.DE", "--date", "2026-09-08", "--register"])

    assert result.exit_code == 0, result.output
    assert "Registered SAP.DE (already registered)" in result.output


def test_analyze_stops_before_any_model_runs_when_the_framework_is_missing(connected, monkeypatch):
    from tradingagents_investboard import _framework

    monkeypatch.setattr(_framework, "missing", lambda: "TradingAgents is not installed. X")

    result = runner.invoke(cli.app, ["analyze", "SAP.DE"])

    assert result.exit_code == 1
    assert "Error: TradingAgents is not installed. X" in result.output


def test_replay_asks_for_the_framework_too(monkeypatch):
    # `replay` imports `.graph`, which imports the framework at module scope.
    from tradingagents_investboard import _framework

    monkeypatch.setattr(_framework, "missing", lambda: "TradingAgents is not installed. X")

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 1
    assert "Error: TradingAgents is not installed. X" in result.output


def test_status_needs_no_framework(connected, monkeypatch):
    from tradingagents_investboard import _framework, client

    def boom():
        raise AssertionError("status must not ask for the framework")

    monkeypatch.setattr(_framework, "require", boom)

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def list_runs(self, ticker):
            return {"items": []}

        def close(self):
            pass

    # `status` imports the class inside the function, so the patch is seen.
    monkeypatch.setattr(client, "InvestboardClient", _Client)

    result = runner.invoke(cli.app, ["status", "SAP.DE"])

    assert result.exit_code == 0
    assert "No runs stored yet." in result.output
