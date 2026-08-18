"""The runtime that ties the pipeline together.

    camera worker -> inference service -> incident engine -> dispatcher
                          |                     |               |
                          v                     v               v
                    ring buffer          snapshot/clip     Twilio + webhooks

The supervisor owns the objects and the wiring; each stage stays independently
testable. It is also the single place that knows the difference between shadow
mode and live alerting.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import time
import zoneinfo

from . import metrics
from .config import CameraConfig, Settings, SiteConfig, load_site_config
from .detect.base import Detector, Frame, FrameResult
from .detect.registry import build_detector
from .detect.service import InferenceService
from .events import EventBus
from .incident.engine import EventKind, IncidentEngine, IncidentEvent
from .ingest.recorder import EvidenceRecorder
from .ingest.worker import CameraWorker, SourceFactory, default_source_factory
from .notify.base import AckBus, CooldownStore, RedisCooldownStore
from .notify.dispatcher import AlertDispatcher, DispatchRun
from .notify.twilio_voice import TwilioSmsChannel, TwilioVoiceChannel
from .notify.webhook import WebhookChannel
from .store import Store

log = logging.getLogger(__name__)


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        site: SiteConfig | None = None,
        detector: Detector | None = None,
        store: Store | None = None,
        source_factory: SourceFactory = default_source_factory,
        channels: dict | None = None,
    ) -> None:
        self.settings = settings
        self.site = site if site is not None else load_site_config(settings.config_path)
        self.bus = EventBus()
        self.store = store or Store(settings.database_url)
        self.ack_bus = AckBus()
        self.engine = IncidentEngine(self.site.cameras)
        self.recorder = EvidenceRecorder(settings.snapshots_dir, settings.clips_dir)
        self.workers: dict[str, CameraWorker] = {}
        self._source_factory = source_factory
        self._detector = detector
        self._inference: InferenceService | None = None
        self._channels = channels
        self.dispatcher: AlertDispatcher | None = None
        self._redis = None
        self._clip_tasks: set[asyncio.Task] = set()
        self._started = False

    # ---- lifecycle --------------------------------------------------------

    @property
    def timezone(self) -> dt.tzinfo | None:
        try:
            return zoneinfo.ZoneInfo(self.site.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            log.warning("unknown timezone %r; using system local time", self.site.timezone)
            return None

    async def start(self) -> None:
        if self._started:
            return
        self.settings.ensure_dirs()
        self.store.create_all()

        detector = self._detector or build_detector(self.settings)
        self._inference = InferenceService(
            detector,
            batch_size=self.settings.batch_size,
            batch_timeout_ms=self.settings.batch_timeout_ms,
        )
        await self._inference.start()

        self.dispatcher = AlertDispatcher(
            site=self.site,
            channels=self._channels or self._build_channels(),
            ack_bus=self.ack_bus,
            cooldowns=await self._build_cooldowns(),
            webhook=WebhookChannel(self.site.alerting.webhooks),
            shadow_mode=self.settings.shadow_mode,
            dashboard_url=self.settings.public_base_url,
            on_run_update=self._on_run_update,
        )

        for camera in self.site.cameras:
            if camera.enabled:
                await self._start_worker(camera)

        self._started = True
        if self.settings.shadow_mode:
            log.warning(
                "FiremeX running in SHADOW MODE -- incidents are recorded but no calls "
                "are placed. Set FIREMEX_SHADOW_MODE=false to go live."
            )
        log.info(
            "supervisor started: %d camera(s), backend=%s",
            len(self.workers),
            detector.name,
        )

    async def stop(self) -> None:
        for worker in list(self.workers.values()):
            await worker.stop()
        self.workers.clear()
        for task in list(self._clip_tasks):
            task.cancel()
        for task in list(self._clip_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._clip_tasks.clear()
        if self.dispatcher is not None:
            await self.dispatcher.shutdown()
        if self._inference is not None:
            await self._inference.stop()
            self._inference = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self.store.dispose()
        self._started = False
        log.info("supervisor stopped")

    def _build_channels(self) -> dict:
        voice = TwilioVoiceChannel(self.settings)
        sms = TwilioSmsChannel(self.settings)
        if not voice.available and not self.settings.shadow_mode:
            log.error(
                "Twilio is not configured but shadow mode is off -- confirmed "
                "incidents will NOT reach anyone by phone."
            )
        return {"call": voice, "sms": sms}

    async def _build_cooldowns(self):
        if not self.settings.redis_url:
            return CooldownStore()
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(self.settings.redis_url, decode_responses=True)
            await client.ping()
        except Exception as exc:  # noqa: BLE001 - Redis is an optimisation, not a requirement
            log.warning("Redis unavailable (%s); using in-memory alert cooldowns", exc)
            return CooldownStore()
        self._redis = client
        log.info("using Redis for alert cooldowns")
        return RedisCooldownStore(client)

    async def _start_worker(self, camera: CameraConfig) -> None:
        if self._inference is None:
            raise RuntimeError("cannot start a camera worker before Supervisor.start()")
        worker = CameraWorker(
            camera=camera,
            submit=self._submit,
            on_result=self._handle_result,
            source_factory=self._source_factory,
            timezone=self.timezone,
        )
        self.workers[camera.id] = worker
        await worker.start()

    async def _submit(self, frame: Frame) -> FrameResult:
        if self._inference is None:
            raise RuntimeError("inference service is not running")
        return await self._inference.submit(frame)

    # ---- the pipeline -----------------------------------------------------

    async def _handle_result(self, result: FrameResult) -> None:
        events = self.engine.observe(result)
        camera_engine = self.engine.get(result.camera_id)
        if camera_engine is not None and result.detections:
            self.bus.publish(
                {
                    "type": "detection",
                    "camera_id": result.camera_id,
                    "wall_ts": result.wall_ts,
                    "detections": [d.as_dict() for d in result.detections],
                    "state": camera_engine.state.value,
                }
            )
        for event in events:
            await self._handle_event(event, result)

    async def _handle_event(self, event: IncidentEvent, result: FrameResult) -> None:
        incident = event.incident
        camera = self.site.camera(incident.camera_id)
        if camera is None:  # pragma: no cover - camera removed mid-flight
            return

        if event.kind is EventKind.OPENED:
            camera_engine = self.engine.get(camera.id)
            await self.store.save_incident(
                incident,
                camera_name=camera.name,
                location=camera.location,
                detections=event.observation.detections,
                assessment=camera_engine.last_assessment if camera_engine else None,
                shadow_mode=self.settings.shadow_mode,
            )
            # Snapshot before dispatch so the SMS link resolves by the time the
            # phone rings.
            await self._capture_snapshot(event, camera)
            self._schedule_clip(event, camera)
            self.bus.publish(self._incident_event_payload("incident.opened", incident, camera))
            if self.dispatcher is not None:
                self.dispatcher.dispatch(incident, camera)

        elif event.kind is EventKind.ESCALATED:
            await self.store.update_incident(
                incident.id,
                severity=incident.severity.value,
                labels=",".join(sorted(incident.labels)),
                growth_ratio=incident.growth_ratio,
                peak_confidence=incident.peak_confidence,
            )
            self.bus.publish(self._incident_event_payload("incident.escalated", incident, camera))

        elif event.kind is EventKind.CLOSED:
            await self.store.close_incident(incident)
            self.ack_bus.forget(incident.id)
            self.bus.publish(self._incident_event_payload("incident.closed", incident, camera))

    async def _capture_snapshot(self, event: IncidentEvent, camera: CameraConfig) -> None:
        worker = self.workers.get(camera.id)
        image = worker.buffer.latest() if worker else None
        if image is None:
            return
        stamp = dt.datetime.fromtimestamp(event.incident.opened_wall).strftime("%Y-%m-%d %H:%M:%S")
        caption = f"{camera.name} - {event.incident.label_summary} - {stamp}"
        path = await self.recorder.capture_snapshot(
            event.incident.id, image, event.observation.detections, caption
        )
        if path:
            event.incident.snapshot_path = path
            await self.store.set_evidence(event.incident.id, snapshot_path=path)

    def _schedule_clip(self, event: IncidentEvent, camera: CameraConfig) -> None:
        worker = self.workers.get(camera.id)
        if worker is None:
            return
        # Pre-roll starts before confirmation, so the clip shows the ignition and
        # not just the aftermath.
        pre_roll_from = event.incident.opened_monotonic - worker.buffer.seconds

        async def _record() -> None:
            path = await self.recorder.capture_clip(
                event.incident.id, worker.buffer, fps=max(worker.observed_fps, 5.0),
                pre_roll_from=pre_roll_from,
            )
            if path:
                event.incident.clip_path = path
                await self.store.set_evidence(event.incident.id, clip_path=path)
                self.bus.publish(
                    {"type": "incident.clip_ready", "incident_id": event.incident.id}
                )

        task = asyncio.create_task(_record(), name=f"firemex-clip-{event.incident.id}")
        self._clip_tasks.add(task)
        task.add_done_callback(self._clip_tasks.discard)

    def _incident_event_payload(self, kind: str, incident, camera: CameraConfig) -> dict:
        return {
            "type": kind,
            "incident_id": incident.id,
            "camera_id": camera.id,
            "camera_name": camera.name,
            "location": camera.location,
            "severity": incident.severity.value,
            "labels": sorted(incident.labels),
            "peak_confidence": round(incident.peak_confidence, 4),
            "growth_ratio": round(incident.growth_ratio, 3),
            "opened_at": incident.opened_wall,
            "shadow_mode": self.settings.shadow_mode,
        }

    async def _on_run_update(self, run: DispatchRun) -> None:
        await self.store.record_run(run)
        self.bus.publish({"type": "alert.update", **run.as_dict()})

    # ---- operator actions -------------------------------------------------

    async def cancel_incident(self, incident_id: str, reason: str = "operator") -> bool:
        """Dismiss an incident: stop escalation and mark it a false positive.

        The one action an operator needs to be able to take instantly, so it does
        both halves -- silencing the calls and recording the verdict that feeds
        fine-tuning -- rather than making them remember two steps.
        """
        camera_engine = self.engine.find_incident(incident_id)
        cancelled_dispatch = (
            self.dispatcher.cancel(incident_id, reason) if self.dispatcher else False
        )
        event = camera_engine.cancel(reason) if camera_engine else None
        if event is not None:
            await self.store.close_incident(event.incident)
        updated = await self.store.update_incident(
            incident_id, review="false_positive", review_note=f"cancelled by {reason}"
        )
        if updated is None and event is None and not cancelled_dispatch:
            return False
        self.bus.publish({"type": "incident.cancelled", "incident_id": incident_id})
        return True

    async def acknowledge_incident(self, incident_id: str, contact_id: str) -> bool:
        if self.dispatcher is not None:
            accepted = self.dispatcher.acknowledge(incident_id, contact_id)
        else:
            # No live dispatcher (already finished, or the API is running without
            # the pipeline). The acknowledgement is still real and still recorded.
            accepted = self.ack_bus.acknowledge(incident_id, contact_id)
            if accepted:
                metrics.ALERTS_ACKNOWLEDGED.inc()
        if accepted:
            await self.store.update_incident(
                incident_id,
                acknowledged_by=contact_id,
                acknowledged_at=dt.datetime.now(dt.UTC),
            )
            self.bus.publish(
                {
                    "type": "incident.acknowledged",
                    "incident_id": incident_id,
                    "contact_id": contact_id,
                }
            )
        return accepted

    async def reload_config(self) -> SiteConfig:
        """Re-read the site YAML and reconcile running workers."""
        site = load_site_config(self.settings.config_path)
        self.site = site
        self.engine.reconfigure(site.cameras)
        if self.dispatcher is not None:
            self.dispatcher.site = site
            self.dispatcher.webhook = WebhookChannel(site.alerting.webhooks)

        wanted = {c.id: c for c in site.cameras if c.enabled}
        for camera_id in list(self.workers):
            worker = self.workers[camera_id]
            replacement = wanted.get(camera_id)
            if replacement is None or replacement != worker.camera:
                await worker.stop()
                del self.workers[camera_id]
        for camera in wanted.values():
            if camera.id in self.workers:
                continue
            if self._inference is None:
                log.debug("not starting worker for %s: pipeline not running", camera.id)
                continue
            await self._start_worker(camera)
        log.info("config reloaded: %d camera(s) running", len(self.workers))
        self.bus.publish({"type": "config.reloaded", "cameras": len(self.workers)})
        return site

    # ---- status -----------------------------------------------------------

    def status(self) -> dict:
        engine_status = {entry["camera_id"]: entry for entry in self.engine.status()}
        cameras = []
        for camera in self.site.cameras:
            worker = self.workers.get(camera.id)
            entry = worker.status() if worker else {
                "camera_id": camera.id,
                "name": camera.name,
                "location": camera.location,
                "connected": False,
                "last_error": "disabled" if not camera.enabled else "not started",
            }
            entry["enabled"] = camera.enabled
            entry["detection"] = engine_status.get(camera.id)
            cameras.append(entry)
        return {
            "site": self.site.name,
            "timezone": self.site.timezone,
            "shadow_mode": self.settings.shadow_mode,
            "detector": self._inference.detector.name if self._inference else None,
            "twilio_configured": self.settings.twilio_configured(),
            "active_incidents": self.engine.active_count(),
            # Full detail, not just a count: a dashboard opened *during* a fire has
            # no incident.opened event to replay, and the cancel button must still
            # be there. That is precisely when someone needs it.
            "incidents": self.open_incidents(),
            "inference_queue": self._inference.pending if self._inference else 0,
            "inference_dropped": self._inference.dropped if self._inference else 0,
            "cameras": cameras,
            "contacts": [
                {"id": c.id, "name": c.name, "channels": c.channels} for c in self.site.contacts
            ],
        }

    def open_incidents(self) -> list[dict]:
        """Every currently-open incident, in the shape the dashboard renders."""
        out = []
        for camera in self.site.cameras:
            engine = self.engine.get(camera.id)
            if engine is None or engine.incident is None:
                continue
            out.append(self._incident_event_payload("incident.open", engine.incident, camera))
        return out

    def healthy(self) -> tuple[bool, list[str]]:
        """Readiness. A camera that is down is a real degradation and says so."""
        problems: list[str] = []
        if not self._started:
            problems.append("supervisor not started")
        for camera in self.site.cameras:
            if not camera.enabled:
                continue
            worker = self.workers.get(camera.id)
            if worker is None:
                problems.append(f"camera {camera.id} has no worker")
                continue
            if not worker.connected:
                problems.append(
                    f"camera {camera.id} disconnected: {worker.last_error or 'unknown'}"
                )
                continue
            # A stream that connected but stopped delivering frames is the
            # dangerous failure: it looks exactly like a quiet room. The watchdog
            # will reconnect it, but readiness must not call it healthy meanwhile.
            last = worker.last_frame_monotonic
            if last is not None:
                age = time.monotonic() - last
                if age > worker.stall_timeout:
                    problems.append(f"camera {camera.id} stalled: no frame for {age:.0f}s")
        if not self.settings.shadow_mode and not self.settings.twilio_configured():
            problems.append("live mode with no Twilio credentials")
        if not self.settings.shadow_mode:
            for camera in self.site.cameras:
                if camera.enabled and not self.site.escalation_chain(camera.id):
                    problems.append(f"camera {camera.id} has no contacts")
        return (not problems, problems)
