"""Alert dispatch and escalation.

One confirmed incident produces exactly one escalation run. The run walks the
camera's contact chain in order, and stops the moment a human acknowledges. That
"stops on acknowledgement" property is the reason the call gathers a digit: a
chain that treats a ringing phone as success will happily leave a fire
unattended.

Guarantees this module is responsible for:

* **De-duplication** -- a ten-minute fire produces one call sequence, not two
  hundred. Enforced with a per-camera cooldown, atomically in Redis when available.
* **A cancel window** -- ``confirm_delay_seconds`` of grace during which an
  operator can dismiss the incident from the dashboard before any phone rings.
* **Escalation** -- per-contact retries, then move down the chain.
* **Shadow mode** -- record and log everything, place no calls. This is the
  default for a new site.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from .. import metrics
from ..config import AlertingConfig, CameraConfig, ContactConfig, SiteConfig
from ..incident.engine import Incident
from .base import AckBus, AlertContext, AlertResult, Channel, CooldownStore, Outcome

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class RunStatus(StrEnum):
    SHADOW = "shadow"
    SUPPRESSED = "suppressed"
    CANCELLED = "cancelled"
    ACKNOWLEDGED = "acknowledged"
    EXHAUSTED = "exhausted"
    NO_CONTACTS = "no_contacts"
    FAILED = "failed"


@dataclass(slots=True)
class DispatchRun:
    """The record of one escalation run, persisted and shown on the dashboard."""

    incident_id: str
    camera_id: str
    status: RunStatus
    attempts: list[AlertResult] = field(default_factory=list)
    acknowledged_by: str | None = None
    started_wall: float = 0.0
    finished_wall: float | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "camera_id": self.camera_id,
            "status": self.status.value,
            "acknowledged_by": self.acknowledged_by,
            "started_wall": self.started_wall,
            "finished_wall": self.finished_wall,
            "detail": self.detail,
            "attempts": [
                {
                    "channel": a.channel,
                    "contact_id": a.contact_id,
                    "outcome": a.outcome.value,
                    "provider_id": a.provider_id,
                    "error": a.error,
                }
                for a in self.attempts
            ],
        }


class AlertDispatcher:
    def __init__(
        self,
        site: SiteConfig,
        channels: dict[str, Channel],
        ack_bus: AckBus | None = None,
        cooldowns: CooldownStore | None = None,
        webhook=None,
        shadow_mode: bool = True,
        dashboard_url: str = "",
        sleep: Sleep = asyncio.sleep,
        on_run_update: Callable[[DispatchRun], Awaitable[None]] | None = None,
    ) -> None:
        self.site = site
        self.channels = channels
        self.ack_bus = ack_bus or AckBus()
        self.cooldowns = cooldowns or CooldownStore()
        self.webhook = webhook
        self.shadow_mode = shadow_mode
        self.dashboard_url = dashboard_url
        self._sleep = sleep
        self._on_run_update = on_run_update
        self._tasks: dict[str, asyncio.Task] = {}
        self.runs: dict[str, DispatchRun] = {}

    # ---- lifecycle --------------------------------------------------------

    def dispatch(self, incident: Incident, camera: CameraConfig) -> asyncio.Task:
        """Start (or return the existing) escalation run for an incident."""
        existing = self._tasks.get(incident.id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(
            self.run(incident, camera), name=f"firemex-alert-{incident.id}"
        )
        self._tasks[incident.id] = task
        return task

    def cancel(self, incident_id: str, reason: str = "operator cancelled") -> bool:
        """Abort an escalation run. Returns True if a run was actually stopped."""
        task = self._tasks.get(incident_id)
        if task is None or task.done():
            return False
        task.cancel()
        run = self.runs.get(incident_id)
        if run is not None:
            run.status = RunStatus.CANCELLED
            run.detail = reason
            run.finished_wall = time.time()
        log.info("escalation for %s cancelled: %s", incident_id, reason)
        return True

    def acknowledge(self, incident_id: str, contact_id: str = "unknown") -> bool:
        """Record a human acknowledgement, stopping the chain at the next check."""
        accepted = self.ack_bus.acknowledge(incident_id, contact_id)
        if accepted:
            metrics.ALERTS_ACKNOWLEDGED.inc()
            log.info("incident %s acknowledged by %s", incident_id, contact_id)
            run = self.runs.get(incident_id)
            if run is not None:
                run.acknowledged_by = contact_id
        return accepted

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    # ---- the escalation run ----------------------------------------------

    async def run(self, incident: Incident, camera: CameraConfig) -> DispatchRun:
        alerting = self.site.alerting
        run = DispatchRun(
            incident_id=incident.id,
            camera_id=camera.id,
            status=RunStatus.SHADOW,
            started_wall=time.time(),
        )
        self.runs[incident.id] = run
        context = self._context(incident, camera)

        try:
            return await self._execute(run, incident, camera, context, alerting)
        except asyncio.CancelledError:
            run.status = RunStatus.CANCELLED
            run.finished_wall = time.time()
            await self._publish(run)
            raise
        except Exception as exc:  # noqa: BLE001 - never let dispatch die silently
            log.exception("escalation for %s failed", incident.id)
            run.status = RunStatus.FAILED
            run.detail = str(exc)
            run.finished_wall = time.time()
            await self._publish(run)
            return run

    async def _execute(
        self,
        run: DispatchRun,
        incident: Incident,
        camera: CameraConfig,
        context: AlertContext,
        alerting: AlertingConfig,
    ) -> DispatchRun:
        # Webhooks fire even in shadow mode: they are how a site wires FiremeX into
        # an existing BMS or Slack, and they carry no risk of a false phone call.
        if self.webhook is not None:
            await self.webhook.broadcast(self._payload(incident, camera, context))

        if self.shadow_mode:
            log.warning(
                "SHADOW MODE: would alert %s for incident %s on %s (%s, severity=%s)",
                [c.id for c in self.site.escalation_chain(camera.id)] or ["<no contacts>"],
                incident.id,
                camera.id,
                context.labels,
                incident.severity,
            )
            run.status = RunStatus.SHADOW
            run.detail = "shadow mode: no calls placed"
            return await self._finish(run)

        cooldown_seconds = alerting.cooldown_minutes * 60.0
        if not await self.cooldowns.should_alert(camera.id, time.time(), cooldown_seconds):
            log.info(
                "suppressing alert for %s: camera %s alerted within the last %.0f min",
                incident.id,
                camera.id,
                alerting.cooldown_minutes,
            )
            run.status = RunStatus.SUPPRESSED
            run.detail = f"within {alerting.cooldown_minutes:g} min cooldown"
            return await self._finish(run)

        chain = self.site.escalation_chain(camera.id)
        if not chain:
            log.error(
                "incident %s on %s has no contacts configured -- nobody will be called",
                incident.id,
                camera.id,
            )
            run.status = RunStatus.NO_CONTACTS
            run.detail = "no contacts configured for this camera"
            return await self._finish(run)

        if alerting.confirm_delay_seconds > 0:
            log.warning(
                "incident %s confirmed on %s; calling in %.0fs unless cancelled",
                incident.id,
                camera.id,
                alerting.confirm_delay_seconds,
            )
            await self._publish(run)
            # Cancellation lands here as CancelledError from the operator's cancel().
            await self._sleep(alerting.confirm_delay_seconds)

        confirmed_at = time.monotonic()
        first_call = True
        for contact in chain:
            acknowledged = await self._try_contact(run, context, contact, first_call, confirmed_at)
            first_call = False
            if acknowledged:
                run.status = RunStatus.ACKNOWLEDGED
                run.acknowledged_by = self.ack_bus.acknowledged_by(incident.id) or contact.id
                return await self._finish(run)
            metrics.ESCALATIONS.inc()
            log.warning(
                "no acknowledgement from %s for %s -- escalating", contact.id, incident.id
            )

        log.critical(
            "incident %s on %s went UNACKNOWLEDGED through the entire chain (%s)",
            incident.id,
            camera.id,
            ", ".join(c.id for c in chain),
        )
        run.status = RunStatus.EXHAUSTED
        run.detail = "chain exhausted with no acknowledgement"
        return await self._finish(run)

    async def _try_contact(
        self,
        run: DispatchRun,
        context: AlertContext,
        contact: ContactConfig,
        send_sms: bool,
        confirmed_at: float,
    ) -> bool:
        alerting = self.site.alerting
        # SMS once per contact, on the first attempt only: repeat texts add noise
        # without adding reach, and the link is identical.
        if "sms" in contact.channels and (sms := self.channels.get("sms")) is not None:
            message = context.render(alerting.sms_template)
            run.attempts.append(await sms.send(context, contact, message))
            await self._publish(run)

        if "call" not in contact.channels:
            return self.ack_bus.is_acknowledged(context.incident_id)

        call = self.channels.get("call")
        if call is None:
            log.error("no voice channel available for contact %s", contact.id)
            return False

        for attempt in range(contact.retries + 1):
            message = context.render(alerting.voice_template)
            result = await call.send(context, contact, message)
            run.attempts.append(result)
            await self._publish(run)
            if send_sms and attempt == 0:
                metrics.ALERT_LATENCY.observe(max(0.0, time.monotonic() - confirmed_at))
            if result.outcome is Outcome.FAILED:
                # A provider failure is not a reason to wait out the ack window.
                log.error(
                    "call attempt %d/%d to %s failed: %s",
                    attempt + 1,
                    contact.retries + 1,
                    contact.id,
                    result.error,
                )
                continue
            if await self.ack_bus.wait(context.incident_id, contact.escalate_after_seconds):
                return True
        return self.ack_bus.is_acknowledged(context.incident_id)

    # ---- helpers ----------------------------------------------------------

    async def _finish(self, run: DispatchRun) -> DispatchRun:
        run.finished_wall = time.time()
        await self._publish(run)
        return run

    async def _publish(self, run: DispatchRun) -> None:
        if self._on_run_update is not None:
            try:
                await self._on_run_update(run)
            except Exception:  # noqa: BLE001 - persistence must not break alerting
                log.exception("failed to publish dispatch run %s", run.incident_id)

    def _context(self, incident: Incident, camera: CameraConfig) -> AlertContext:
        base = self.dashboard_url.rstrip("/")
        return AlertContext(
            incident_id=incident.id,
            camera_id=camera.id,
            camera_name=camera.name,
            location=camera.location,
            site_name=self.site.name,
            labels=incident.label_summary,
            severity=incident.severity.value,
            snapshot_url=(
                f"{base}/api/incidents/{incident.id}/snapshot"
                if base and incident.snapshot_path
                else None
            ),
            dashboard_url=f"{base}/#/incidents/{incident.id}" if base else None,
        )

    def _payload(self, incident: Incident, camera: CameraConfig, context: AlertContext) -> dict:
        return {
            "event": "incident.confirmed",
            "incident_id": incident.id,
            "camera": {"id": camera.id, "name": camera.name, "location": camera.location},
            "site": self.site.name,
            "severity": incident.severity.value,
            "labels": sorted(incident.labels),
            "peak_confidence": round(incident.peak_confidence, 4),
            "growth_ratio": round(incident.growth_ratio, 3),
            "opened_at": incident.opened_wall,
            "shadow_mode": self.shadow_mode,
            "snapshot_url": context.snapshot_url,
            "dashboard_url": context.dashboard_url,
        }
