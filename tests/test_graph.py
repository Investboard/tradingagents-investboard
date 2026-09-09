import json
import os
import stat

import httpx
import pytest

from tradingagents_investboard import graph
from tradingagents_investboard.client import InvestboardApiError


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "OUTBOX_DIR", tmp_path / "investboard" / "outbox")
    monkeypatch.setattr(graph, "access_token", lambda *args, **kwargs: "tok")
    monkeypatch.setattr(graph.time, "sleep", lambda seconds: None)
    return tmp_path / "investboard" / "outbox"


def counting_transport(response_for):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response_for(request)

    return httpx.MockTransport(handler), calls


def test_an_outbox_entry_is_owner_only(outbox):
    """A failed post parks the whole run payload on disk, so the file, the outbox
    and the directory holding it are all owner-only."""
    path = graph.write_outbox({"framework_run_id": "run-1", "ticker": "SAP.DE"})

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path.parent.parent).st_mode) == 0o700
    assert json.loads(path.read_text(encoding="utf-8"))["framework_run_id"] == "run-1"


def test_a_hostile_run_id_cannot_escape_the_outbox(outbox):
    """The run id is built from a ticker the model echoed back, so it is not a path."""
    path = graph.write_outbox({"framework_run_id": "../../etc/passwd"})

    assert path.parent == outbox
    assert path.name == ".._.._etc_passwd.json"


def test_the_written_entry_leaves_no_temporary_behind(outbox):
    graph.write_outbox({"framework_run_id": "run-1"})

    assert sorted(p.name for p in outbox.iterdir()) == ["run-1.json"]


def test_a_400_is_not_retried(outbox):
    """A malformed payload is malformed on every attempt; retrying only delays the message."""
    transport, calls = counting_transport(
        lambda request: httpx.Response(
            400, json={"error": {"message": "bad", "details": {"reason": "invalid_payload"}}}
        )
    )

    with pytest.raises(InvestboardApiError) as excinfo:
        graph.post_payload({"framework_run_id": "run-1"}, transport=transport)

    assert excinfo.value.status == 400
    assert len(calls) == 1


def test_a_503_is_retried_three_times(outbox):
    transport, calls = counting_transport(lambda request: httpx.Response(503, text="down"))

    with pytest.raises(InvestboardApiError) as excinfo:
        graph.post_payload({"framework_run_id": "run-1"}, transport=transport)

    assert excinfo.value.status == 503
    assert len(calls) == 3


@pytest.mark.parametrize("status", graph.PERMANENT_STATUSES)
def test_replay_sets_a_permanent_refusal_aside_and_keeps_going(outbox, status):
    """One entry Investboard will never accept must not block the ones behind it."""
    graph.write_outbox({"framework_run_id": "run-a", "ticker": "NOPE"})
    graph.write_outbox({"framework_run_id": "run-b", "ticker": "SAP.DE"})

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        if payload["framework_run_id"] == "run-a":
            details = {"reason": "subject_unresolvable", "field": "ticker"}
            body = {"error": {"message": "no subject", "details": details}}
            return httpx.Response(status, json=body)
        return httpx.Response(201, json={"data": {"id": "stored-1"}})

    transport, calls = counting_transport(respond)

    outcome = graph.replay_outbox(transport=transport)

    assert outcome == {"sent": ["run-b"], "rejected": ["run-a"], "failed": []}
    assert len(calls) == 2
    assert sorted(p.name for p in outbox.iterdir()) == ["run-a.json.rejected"]


@pytest.mark.parametrize("status", graph.REQUEUE_STATUSES)
def test_replay_keeps_a_requeue_refusal_in_the_queue(outbox, status):
    """A lapsed connection, a paused subscription and a spent daily cap refuse
    the account, not the run.

    The payload is good and will be accepted once the account is, so it keeps
    its place in the queue. Setting it aside as `.rejected` made the user
    rename a file by hand to recover a run nothing was wrong with.
    """
    graph.write_outbox({"framework_run_id": "run-a"})
    graph.write_outbox({"framework_run_id": "run-b"})

    transport, calls = counting_transport(
        lambda request: httpx.Response(status, json={"error": {"message": "not now"}})
    )

    outcome = graph.replay_outbox(transport=transport)

    assert outcome == {"sent": [], "rejected": [], "failed": ["run-a"]}
    # Not retried inside the post, and the entry behind it is not tried at all.
    assert len(calls) == 1
    assert sorted(p.name for p in outbox.iterdir()) == ["run-a.json", "run-b.json"]


def test_a_second_refusal_does_not_overwrite_the_first(outbox):
    """Two runs can carry the same id, and the outbox throws nothing away."""
    refused = httpx.MockTransport(
        lambda request: httpx.Response(422, json={"error": {"message": "no subject"}})
    )
    graph.write_outbox({"framework_run_id": "run-a", "attempt": 1})
    graph.replay_outbox(transport=refused)
    graph.write_outbox({"framework_run_id": "run-a", "attempt": 2})

    graph.replay_outbox(transport=refused)

    names = sorted(p.name for p in outbox.iterdir())
    assert len(names) == 2
    assert "run-a.json.rejected" in names
    assert all(name.endswith(".rejected") for name in names)


def test_replay_stops_on_a_transient_failure_and_leaves_the_file(outbox):
    graph.write_outbox({"framework_run_id": "run-a"})
    graph.write_outbox({"framework_run_id": "run-b"})

    transport, calls = counting_transport(lambda request: httpx.Response(503, text="down"))

    outcome = graph.replay_outbox(transport=transport)

    assert outcome == {"sent": [], "rejected": [], "failed": ["run-a"]}
    # Three attempts at the first entry, then it stops: the second is untouched.
    assert len(calls) == 3
    assert sorted(p.name for p in outbox.iterdir()) == ["run-a.json", "run-b.json"]


def test_replay_of_an_empty_outbox_reports_nothing(outbox):
    assert graph.replay_outbox() == {"sent": [], "rejected": [], "failed": []}


def test_the_instrument_context_carries_the_two_blocks(monkeypatch):
    """The framework's paragraph first, then the owner's policy and position.

    The base method is patched on the framework class the override calls
    through ``super()``, so no yfinance lookup runs; ``__new__`` skips the
    eager LLM construction in ``__init__``, which this test must not run.
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    from tradingagents_investboard import context as context_module

    seen: dict[str, object] = {}

    def blocks(client, ticker):
        seen["client"] = client
        seen["ticker"] = ticker
        return "\n\nPOLICY\n\nPOSITION"

    monkeypatch.setattr(graph, "access_token", lambda *args, **kwargs: "tok")
    monkeypatch.setattr(graph, "base_url", lambda: "https://app.example")
    monkeypatch.setattr(
        TradingAgentsGraph,
        "resolve_instrument_context",
        lambda self, ticker, asset_type="stock": f"The instrument to analyze is `{ticker}`.",
    )
    monkeypatch.setattr(context_module, "instrument_context_blocks", blocks)

    instance = graph.InvestboardTradingAgentsGraph.__new__(graph.InvestboardTradingAgentsGraph)
    text = instance.resolve_instrument_context("SAP.DE")

    assert text == "The instrument to analyze is `SAP.DE`.\n\nPOLICY\n\nPOSITION"
    assert seen["ticker"] == "SAP.DE"
    # The client holds a connection pool and is used once, so the run must not
    # leave it open behind the graph.
    assert seen["client"].is_closed
