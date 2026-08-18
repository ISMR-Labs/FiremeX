"""Notification channel contracts and the acknowledgement bus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Outcome(StrEnum):
    QUEUED = "queued"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class AlertContext:
    """Everything a channel needs to render an alert."""

    incident_id: str
    camera_id: str
    camera_name: str
    location: str
    site_name: str
    labels: str
    severity: str
    snapshot_url: str | None = None
    dashboard_url: str | None = None

    def render(self, template: str) -> str:
        return template.format(
            site=self.site_name,
            camera=self.camera_name,
            camera_id=self.camera_id,
            location=self.location or "location not set",
            labels=self.labels,
            severity=self.severity,
            link=self.snapshot_url or self.dashboard_url or "",
        )


@dataclass(slots=True)
class AlertResult:
    channel: str
    contact_id: str
    outcome: Outcome
    provider_id: str | None = None
    error: str | None = None


@runtime_checkable
class Channel(Protocol):
    name: str

    async def send(self, context: AlertContext, contact, message: str) -> AlertResult: ...


class AckBus:
    """Routes acknowledgements from the inbound webhook back to the waiting
    escalation loop.

    Keyed by incident, so any contact in the chain can acknowledge and stop it. The
    ack may arrive before the loop starts waiting -- a fast responder pressing 1
    while the next call is still being placed -- so acknowledgements are recorded
    even with no waiter, and ``wait`` returns immediately for an already-acked
    incident.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._acked_by: dict[str, str] = {}

    def _event(self, incident_id: str) -> asyncio.Event:
        return self._events.setdefault(incident_id, asyncio.Event())

    def acknowledge(self, incident_id: str, contact_id: str = "unknown") -> bool:
        """Record an acknowledgement. Returns False if already acknowledged."""
        if incident_id in self._acked_by:
            return False
        self._acked_by[incident_id] = contact_id
        self._event(incident_id).set()
        return True

    def acknowledged_by(self, incident_id: str) -> str | None:
        return self._acked_by.get(incident_id)

    def is_acknowledged(self, incident_id: str) -> bool:
        return incident_id in self._acked_by

    async def wait(self, incident_id: str, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for an acknowledgement."""
        if self.is_acknowledged(incident_id):
            return True
        try:
            await asyncio.wait_for(self._event(incident_id).wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def forget(self, incident_id: str) -> None:
        self._events.pop(incident_id, None)
        self._acked_by.pop(incident_id, None)


@dataclass(slots=True)
class CooldownStore:
    """Per-camera alert suppression.

    In-memory by default; :class:`RedisCooldownStore` is used when Redis is
    configured so suppression survives a restart and is shared across processes.
    """

    _seen: dict[str, float] = field(default_factory=dict)

    async def should_alert(self, camera_id: str, now: float, cooldown_seconds: float) -> bool:
        last = self._seen.get(camera_id)
        if last is not None and now - last < cooldown_seconds:
            return False
        self._seen[camera_id] = now
        return True

    async def clear(self, camera_id: str) -> None:
        self._seen.pop(camera_id, None)


class RedisCooldownStore:
    """Cooldown backed by Redis SET NX EX, so suppression is atomic across workers."""

    def __init__(self, client) -> None:
        self._client = client

    async def should_alert(self, camera_id: str, now: float, cooldown_seconds: float) -> bool:
        key = f"firemex:cooldown:{camera_id}"
        acquired = await self._client.set(key, str(now), nx=True, ex=int(max(1, cooldown_seconds)))
        return bool(acquired)

    async def clear(self, camera_id: str) -> None:
        await self._client.delete(f"firemex:cooldown:{camera_id}")
