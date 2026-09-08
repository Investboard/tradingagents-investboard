import pytest
from typer.testing import CliRunner

from tradingagents_investboard import cli

runner = CliRunner()


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


def test_a_ticker_that_is_not_a_ticker_is_refused(connected):
    result = runner.invoke(cli.app, ["analyze", "SAP DE; rm -rf /"])

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
    import tradingagents_investboard.graph as graph

    monkeypatch.setattr(
        graph, "replay_outbox", lambda: {"sent": ["a"], "rejected": ["b"], "failed": []}
    )

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 1
    assert "Sent 1, rejected 1, still queued 0." in result.output
    assert "Rejected: b" in result.output


def test_replay_of_an_empty_outbox_succeeds(monkeypatch):
    import tradingagents_investboard.graph as graph

    monkeypatch.setattr(
        graph, "replay_outbox", lambda: {"sent": [], "rejected": [], "failed": []}
    )

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 0
    assert "Outbox is empty." in result.output
