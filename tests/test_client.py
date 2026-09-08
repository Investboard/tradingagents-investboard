import httpx
import pytest

from tradingagents_investboard.client import InvestboardApiError, InvestboardClient


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return InvestboardClient("https://app.example", "tok", transport=transport)


def test_post_run_sends_bearer_and_idempotency_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        seen["path"] = request.url.path
        return httpx.Response(
            201, json={"data": {"id": "r1"}, "error": None, "meta": {"idempotent": False}}
        )

    client = make_client(handler)
    result = client.post_run(
        {"framework_run_id": "SAP.DE_2026-09-08_abc12345", "ticker": "SAP.DE"}
    )
    assert result == {"id": "r1"}
    assert seen == {
        "auth": "Bearer tok",
        "idem": "SAP.DE_2026-09-08_abc12345",
        "path": "/api/v1/agent/runs",
    }


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "unauthorized"),
        (402, "access_paused"),
        (413, "payload_too_large"),
        (422, "subject_unresolvable"),
        (429, "daily_cap_reached"),
    ],
)
def test_post_run_maps_refusals(status, code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "data": None,
                "error": {"code": "X", "message": "m", "details": {"reason": code}},
                "meta": {},
            },
        )

    client = make_client(handler)
    with pytest.raises(InvestboardApiError) as excinfo:
        client.post_run({"framework_run_id": "SAP.DE_2026-09-08_abc12345"})
    assert excinfo.value.status == status
    assert excinfo.value.reason == code


def test_list_runs_builds_the_query():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["ticker"] == "SAP.DE"
        assert request.url.params["limit"] == "5"
        return httpx.Response(
            200, json={"data": {"items": [], "next_before": None}, "error": None, "meta": {}}
        )

    assert make_client(handler).list_runs("SAP.DE") == {"items": [], "next_before": None}
