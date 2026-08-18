"""Webhook channel -- fan incidents out to whatever else the site runs."""

from __future__ import annotations

import logging

import httpx

from .. import metrics
from .base import AlertContext, AlertResult, Outcome

log = logging.getLogger(__name__)


class WebhookChannel:
    name = "webhook"

    def __init__(
        self, urls: list[str], timeout: float = 5.0, client: httpx.AsyncClient | None = None
    ) -> None:
        self.urls = list(urls)
        self.timeout = timeout
        self._client = client

    async def broadcast(self, payload: dict) -> list[AlertResult]:
        if not self.urls:
            return []
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        results: list[AlertResult] = []
        try:
            for url in self.urls:
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    metrics.ALERTS_SENT.labels(channel=self.name, outcome="queued").inc()
                    results.append(
                        AlertResult(channel=self.name, contact_id=url, outcome=Outcome.QUEUED)
                    )
                except Exception as exc:  # noqa: BLE001 - a dead webhook must not block alerting
                    log.warning("webhook %s failed: %s", url, exc)
                    metrics.ALERTS_SENT.labels(channel=self.name, outcome="failed").inc()
                    results.append(
                        AlertResult(
                            channel=self.name,
                            contact_id=url,
                            outcome=Outcome.FAILED,
                            error=str(exc),
                        )
                    )
        finally:
            if self._client is None:
                await client.aclose()
        return results

    async def send(self, context: AlertContext, contact, message: str) -> AlertResult:
        results = await self.broadcast({"incident_id": context.incident_id, "message": message})
        return results[0] if results else AlertResult(self.name, "none", Outcome.SKIPPED)
