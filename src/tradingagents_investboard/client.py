"""Thin HTTP client for the Investboard agent endpoints."""

from __future__ import annotations

from typing import Any

import httpx

REASON_BY_STATUS = {
    401: "unauthorized",
    402: "access_paused",
    413: "payload_too_large",
    422: "subject_unresolvable",
    429: "daily_cap_reached",
}


class InvestboardApiError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(f"{status} {reason}: {message}")
        self.status = status
        self.reason = reason
        self.message = message


class InvestboardClient:
    def __init__(
        self,
        base_url: str,
        access_token: str,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {access_token}", "accept": "application/json"},
            timeout=httpx.Timeout(30.0, read=60.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _unwrap(self, response: httpx.Response) -> Any:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            error = (body or {}).get("error") or {}
            details = error.get("details") or {}
            reason = details.get("reason") or REASON_BY_STATUS.get(response.status_code, "error")
            raise InvestboardApiError(
                response.status_code, reason, error.get("message") or response.text[:200]
            )
        return body.get("data")

    def post_run(self, payload: dict[str, Any]) -> Any:
        response = self._client.post(
            "/api/v1/agent/runs",
            json=payload,
            headers={"idempotency-key": str(payload.get("framework_run_id", ""))},
        )
        return self._unwrap(response)

    def list_runs(self, ticker: str, limit: int = 5) -> Any:
        response = self._client.get(
            "/api/v1/agent/runs", params={"ticker": ticker, "limit": str(limit)}
        )
        return self._unwrap(response)
