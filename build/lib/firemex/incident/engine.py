"""Temporal confirmation: turning per-frame detections into incidents.

This module, not the model, is where a usable fire detector is won or lost. A
detector that cries wolf gets unplugged, and then it protects nothing. Every rule
here exists to reject a specific real-world false positive:

filtering
    Per-class, day/night confidence floors, exclusion zones, and a minimum box
    area. Kills the long tail of tiny low-confidence noise and known-bad regions.
persistence
    A detection must recur across most of a sliding window of sampled frames.
    Kills single-frame flukes, compression artefacts, and sensor noise.
spatial stability
    Those recurring boxes must overlap each other. A real fire stays put; a
    passing headlight or hi-vis vest sweeps across the frame.
growth
    Fire and smoke grow. A detection whose area is collapsing is treated as
    flicker, and sustained growth escalates severity.

The engine is pure: it holds no I/O, no clock and no database. Time arrives with
each frame as a monotonic timestamp, which makes the whole state machine
deterministic and directly testable, and means an NTP step cannot corrupt it.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean

from .. import metrics
from ..config import CameraConfig
from ..detect.base import FIRE, BBox, Detection, FrameResult, iou
from .zones import ZoneMask

log = logging.getLogger(__name__)


class State(StrEnum):
    IDLE = "idle"
    #: Something is being detected but has not yet satisfied the confirmation rules.
    CANDIDATE = "candidate"
    #: Confirmed. An incident is open and alerting has been handed to the dispatcher.
    CONFIRMED = "confirmed"


class Severity(StrEnum):
    #: Smoke only, or a small stable detection. Still alerts -- smoke first is the
    #: normal way a real fire presents, and the whole point is early warning.
    WARNING = "warning"
    #: Visible flame, or a detection growing fast.
    CRITICAL = "critical"


class Reason(StrEnum):
    BELOW_THRESHOLD = "below_threshold"
    EXCLUDED_ZONE = "excluded_zone"
    TOO_SMALL = "too_small"
    UNKNOWN_LABEL = "unknown_label"
    INSUFFICIENT_PERSISTENCE = "insufficient_persistence"
    NOT_SPATIALLY_STABLE = "not_spatially_stable"
    SHRINKING = "shrinking"


@dataclass(frozen=True, slots=True)
class Observation:
    """One sampled frame after filtering."""

    monotonic_ts: float
    wall_ts: float
    detections: tuple[Detection, ...]

    @property
    def hit(self) -> bool:
        return bool(self.detections)

    @property
    def area(self) -> float:
        """Union area of this frame's boxes.

        Union rather than sum, so two overlapping boxes over one fire are not
        counted as twice the fire.
        """
        return _union_area(d.box for d in self.detections)

    @property
    def envelope(self) -> BBox | None:
        boxes = [d.box for d in self.detections]
        if not boxes:
            return None
        envelope = boxes[0]
        for box in boxes[1:]:
            envelope = envelope.union(box)
        return envelope


@dataclass(slots=True)
class Incident:
    id: str
    camera_id: str
    opened_monotonic: float
    opened_wall: float
    severity: Severity
    labels: set[str]
    peak_confidence: float
    envelope: BBox | None
    frames_confirmed: int
    growth_ratio: float = 1.0
    last_hit_monotonic: float = 0.0
    closed_wall: float | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None

    @property
    def label_summary(self) -> str:
        if {"fire", "smoke"} <= self.labels:
            return "Fire and smoke"
        if FIRE in self.labels:
            return "Fire"
        return "Smoke"


class EventKind(StrEnum):
    OPENED = "opened"
    ESCALATED = "escalated"
    CLOSED = "closed"


@dataclass(slots=True)
class IncidentEvent:
    kind: EventKind
    #: The live incident object. Mutable by design -- the dispatcher wants current
    #: state -- so anything that inspects an event after the fact must use the
    #: snapshot fields below rather than reading through to the incident.
    incident: Incident
    observation: Observation
    #: Severity and labels as they were when this event was emitted.
    severity: Severity = Severity.WARNING
    labels: frozenset[str] = frozenset()

    @classmethod
    def of(cls, kind: EventKind, incident: Incident, observation: Observation) -> IncidentEvent:
        return cls(
            kind=kind,
            incident=incident,
            observation=observation,
            severity=incident.severity,
            labels=frozenset(incident.labels),
        )


@dataclass(slots=True)
class Assessment:
    """Why the engine did or did not confirm. Surfaced to the dashboard and logs so
    "why didn't it call?" always has an answer."""

    confirmed: bool
    hit_frames: int
    frames_required: int
    cluster_size: int
    stability_required: int
    growth_ratio: float
    rejected: Reason | None = None


def _union_area(boxes: Iterable[BBox]) -> float:
    """Area of the union of axis-aligned boxes, via coordinate-grid decomposition.

    Exact, and the box count per frame is tiny, so the O(n^2) grid is free.
    """
    boxes = [b for b in boxes if b.area > 0]
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return boxes[0].area
    xs = sorted({b.x1 for b in boxes} | {b.x2 for b in boxes})
    ys = sorted({b.y1 for b in boxes} | {b.y2 for b in boxes})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx = (xs[i] + xs[i + 1]) / 2
            cy = (ys[j] + ys[j + 1]) / 2
            if any(b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2 for b in boxes):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total


def largest_stable_cluster(boxes: list[BBox], threshold: float) -> int:
    """Size of the biggest group of boxes that all overlap one common box.

    A cheap stand-in for tracking: if N frames produced boxes that mutually
    overlap, the thing being detected has stayed in one place for N frames.
    """
    if not boxes:
        return 0
    return max(sum(1 for other in boxes if iou(seed, other) >= threshold) for seed in boxes)


def _default_id_factory(camera_id: str) -> str:
    return f"{camera_id}-{uuid.uuid4().hex[:10]}"


class CameraIncidentEngine:
    """Confirmation state machine for a single camera."""

    def __init__(
        self,
        camera: CameraConfig,
        id_factory: Callable[[str], str] = _default_id_factory,
    ) -> None:
        self.camera = camera
        self._id_factory = id_factory
        self._zones = ZoneMask(camera.exclude_zones)
        self._window: deque[Observation] = deque(maxlen=camera.confirm.window)
        #: Longer (ts, area) history for the severity growth trend, pruned by time.
        self._area_history: deque[tuple[float, float]] = deque()
        self.state = State.IDLE
        self.incident: Incident | None = None
        self.last_assessment: Assessment | None = None
        self.last_hit_wall: float | None = None
        self.frames_seen = 0
        #: Incidents cancelled by an operator; suppresses re-opening on the same event.
        self._suppress_until_clear = False

    # ---- public API -------------------------------------------------------

    def observe(self, result: FrameResult) -> list[IncidentEvent]:
        """Feed one detector result. Returns any incident lifecycle events."""
        self.frames_seen += 1
        observation = self._filter(result)
        self._window.append(observation)
        self._record_area(observation)

        if observation.hit:
            self.last_hit_wall = observation.wall_ts
            for detection in observation.detections:
                metrics.DETECTIONS.labels(camera=self.camera.id, label=detection.label).inc()

        if self.state is State.CONFIRMED:
            return self._observe_confirmed(observation)
        return self._observe_unconfirmed(observation)

    def cancel(self, reason: str = "operator") -> IncidentEvent | None:
        """Cancel the open incident.

        Suppresses re-confirmation until the camera goes quiet, so an operator who
        dismisses a sunset is not re-alerted three frames later.
        """
        if self.incident is None:
            return None
        incident = self.incident
        log.info("incident %s cancelled on %s (%s)", incident.id, self.camera.id, reason)
        metrics.INCIDENTS_CANCELLED.labels(camera=self.camera.id).inc()
        self._suppress_until_clear = True
        return self._close(reason=reason)

    def status(self) -> dict:
        recent = list(self._window)
        return {
            "camera_id": self.camera.id,
            "state": self.state.value,
            "frames_seen": self.frames_seen,
            "window_hits": sum(1 for obs in recent if obs.hit),
            "window_size": len(recent),
            "frames_required": self.camera.confirm.frames_required,
            "last_hit_wall": self.last_hit_wall,
            "incident_id": self.incident.id if self.incident else None,
            "severity": self.incident.severity.value if self.incident else None,
            "assessment": (
                {
                    "confirmed": self.last_assessment.confirmed,
                    "hit_frames": self.last_assessment.hit_frames,
                    "cluster_size": self.last_assessment.cluster_size,
                    "growth_ratio": round(self.last_assessment.growth_ratio, 3),
                    "rejected": (
                        self.last_assessment.rejected.value
                        if self.last_assessment.rejected
                        else None
                    ),
                }
                if self.last_assessment
                else None
            ),
        }

    # ---- filtering --------------------------------------------------------

    def _filter(self, result: FrameResult) -> Observation:
        thresholds = self.camera.thresholds.select(result.is_night)
        confirm = self.camera.confirm
        kept: list[Detection] = []
        for detection in result.detections:
            reason = None
            if detection.label not in ("fire", "smoke"):
                reason = Reason.UNKNOWN_LABEL
            elif detection.confidence < thresholds.for_label(detection.label):
                reason = Reason.BELOW_THRESHOLD
            elif detection.box.area < confirm.min_box_area:
                reason = Reason.TOO_SMALL
            elif self._zones.excludes(detection.box):
                reason = Reason.EXCLUDED_ZONE
            if reason is not None:
                metrics.DETECTIONS_SUPPRESSED.labels(
                    camera=self.camera.id, reason=reason.value
                ).inc()
                continue
            kept.append(detection)
        return Observation(
            monotonic_ts=result.monotonic_ts,
            wall_ts=result.wall_ts,
            detections=tuple(kept),
        )

    def _record_area(self, observation: Observation) -> None:
        area = observation.area if observation.hit else 0.0
        self._area_history.append((observation.monotonic_ts, area))
        cutoff = observation.monotonic_ts - self.camera.confirm.growth_window_seconds
        while self._area_history and self._area_history[0][0] < cutoff:
            self._area_history.popleft()

    # ---- state transitions ------------------------------------------------

    def _observe_unconfirmed(self, observation: Observation) -> list[IncidentEvent]:
        hits = [obs for obs in self._window if obs.hit]
        self.state = State.CANDIDATE if hits else State.IDLE

        if not hits:
            # Camera is quiet again: lift any operator cancellation.
            self._suppress_until_clear = False

        assessment = self._assess(hits)
        self.last_assessment = assessment

        if not assessment.confirmed or self._suppress_until_clear:
            return []
        return [self._open(observation, assessment)]

    def _observe_confirmed(self, observation: Observation) -> list[IncidentEvent]:
        incident = self.incident
        assert incident is not None  # invariant of State.CONFIRMED
        if observation.hit:
            incident.last_hit_monotonic = observation.monotonic_ts
            incident.labels |= {d.label for d in observation.detections}
            incident.peak_confidence = max(
                incident.peak_confidence, max(d.confidence for d in observation.detections)
            )
            envelope = observation.envelope
            if envelope is not None:
                incident.envelope = (
                    envelope if incident.envelope is None else incident.envelope.union(envelope)
                )
            incident.growth_ratio = self._growth_ratio()
            severity = self._severity(incident.labels, incident.growth_ratio)
            if severity is Severity.CRITICAL and incident.severity is not Severity.CRITICAL:
                incident.severity = severity
                log.warning(
                    "incident %s escalated to critical on %s (labels=%s growth=%.2f)",
                    incident.id,
                    self.camera.id,
                    sorted(incident.labels),
                    incident.growth_ratio,
                )
                return [IncidentEvent.of(EventKind.ESCALATED, incident, observation)]
            return []

        idle_for = observation.monotonic_ts - incident.last_hit_monotonic
        if idle_for >= self.camera.confirm.clear_after_seconds:
            event = self._close(reason="cleared", observation=observation)
            return [event] if event else []
        return []

    def _open(self, observation: Observation, assessment: Assessment) -> IncidentEvent:
        labels = {d.label for d in observation.detections}
        # The confirming frame can lag the window; take labels from the whole window
        # so a fire seen two frames ago still counts.
        for obs in self._window:
            labels |= {d.label for d in obs.detections}
        growth_ratio = self._growth_ratio()
        confidences = [d.confidence for obs in self._window for d in obs.detections]
        incident = Incident(
            id=self._id_factory(self.camera.id),
            camera_id=self.camera.id,
            opened_monotonic=observation.monotonic_ts,
            opened_wall=observation.wall_ts,
            severity=self._severity(labels, growth_ratio),
            labels=labels,
            peak_confidence=max(confidences, default=0.0),
            envelope=self._window_envelope(),
            frames_confirmed=assessment.hit_frames,
            growth_ratio=growth_ratio,
            last_hit_monotonic=observation.monotonic_ts,
        )
        self.incident = incident
        self.state = State.CONFIRMED
        metrics.INCIDENTS_OPENED.labels(camera=self.camera.id).inc()
        log.warning(
            "INCIDENT OPENED %s camera=%s labels=%s severity=%s frames=%d/%d growth=%.2f",
            incident.id,
            self.camera.id,
            sorted(labels),
            incident.severity.value,
            assessment.hit_frames,
            assessment.frames_required,
            growth_ratio,
        )
        return IncidentEvent.of(EventKind.OPENED, incident, observation)

    def _close(self, reason: str, observation: Observation | None = None) -> IncidentEvent | None:
        incident = self.incident
        if incident is None:
            return None
        last = observation or (self._window[-1] if self._window else None)
        incident.closed_wall = last.wall_ts if last else incident.opened_wall
        self.incident = None
        self.state = State.CANDIDATE if (last and last.hit) else State.IDLE
        log.info("incident %s closed on %s (%s)", incident.id, self.camera.id, reason)
        return IncidentEvent.of(
            EventKind.CLOSED,
            incident,
            last or Observation(0.0, incident.opened_wall, ()),
        )

    # ---- the confirmation rules ------------------------------------------

    def _assess(self, hits: list[Observation]) -> Assessment:
        confirm = self.camera.confirm
        required = confirm.frames_required
        stability_required = confirm.stability_frames or required
        growth_ratio = self._growth_ratio()

        if len(hits) < required:
            return Assessment(
                confirmed=False,
                hit_frames=len(hits),
                frames_required=required,
                cluster_size=0,
                stability_required=stability_required,
                growth_ratio=growth_ratio,
                rejected=Reason.INSUFFICIENT_PERSISTENCE,
            )

        envelopes = [obs.envelope for obs in hits]
        boxes = [box for box in envelopes if box is not None]
        cluster = largest_stable_cluster(boxes, confirm.stability_iou)
        if cluster < stability_required:
            metrics.DETECTIONS_SUPPRESSED.labels(
                camera=self.camera.id, reason=Reason.NOT_SPATIALLY_STABLE.value
            ).inc()
            return Assessment(
                confirmed=False,
                hit_frames=len(hits),
                frames_required=required,
                cluster_size=cluster,
                stability_required=stability_required,
                growth_ratio=growth_ratio,
                rejected=Reason.NOT_SPATIALLY_STABLE,
            )

        if confirm.require_growth and not self._area_not_shrinking(hits):
            metrics.DETECTIONS_SUPPRESSED.labels(
                camera=self.camera.id, reason=Reason.SHRINKING.value
            ).inc()
            return Assessment(
                confirmed=False,
                hit_frames=len(hits),
                frames_required=required,
                cluster_size=cluster,
                stability_required=stability_required,
                growth_ratio=growth_ratio,
                rejected=Reason.SHRINKING,
            )

        return Assessment(
            confirmed=True,
            hit_frames=len(hits),
            frames_required=required,
            cluster_size=cluster,
            stability_required=stability_required,
            growth_ratio=growth_ratio,
        )

    def _area_not_shrinking(self, hits: list[Observation]) -> bool:
        """Cheap flicker filter over the confirm window.

        Only a few seconds of history is available at confirmation time, so this is
        deliberately permissive -- it rejects a collapsing detection, not a merely
        flat one. The strong growth signal needs 20-30 s and is used for severity
        escalation instead, because waiting that long before alerting would defeat
        the point of early detection.
        """
        areas = [obs.area for obs in hits]
        if len(areas) < 4:
            return True
        split = len(areas) // 2
        first = fmean(areas[:split])
        second = fmean(areas[split:])
        if first <= 0:
            return True
        return second >= first * (1.0 - self.camera.confirm.growth_tolerance)

    def _growth_ratio(self) -> float:
        """Area trend over the long window: >1 growing, <1 shrinking."""
        samples = [area for _, area in self._area_history if area > 0]
        if len(samples) < 6:
            return 1.0
        third = max(2, len(samples) // 3)
        early = fmean(samples[:third])
        late = fmean(samples[-third:])
        if early <= 0:
            return 1.0
        return late / early

    def _severity(self, labels: set[str], growth_ratio: float) -> Severity:
        if FIRE in labels:
            return Severity.CRITICAL
        # Smoke alone is a warning until it starts growing quickly, at which point
        # it is behaving like a developing fire.
        if growth_ratio >= 1.6:
            return Severity.CRITICAL
        return Severity.WARNING

    def _window_envelope(self) -> BBox | None:
        envelope: BBox | None = None
        for obs in self._window:
            box = obs.envelope
            if box is None:
                continue
            envelope = box if envelope is None else envelope.union(box)
        return envelope


class IncidentEngine:
    """Owns one :class:`CameraIncidentEngine` per camera."""

    def __init__(
        self,
        cameras: Iterable[CameraConfig] = (),
        id_factory: Callable[[str], str] = _default_id_factory,
    ) -> None:
        self._id_factory = id_factory
        self._engines: dict[str, CameraIncidentEngine] = {}
        for camera in cameras:
            self.add_camera(camera)

    def add_camera(self, camera: CameraConfig) -> CameraIncidentEngine:
        engine = CameraIncidentEngine(camera, id_factory=self._id_factory)
        self._engines[camera.id] = engine
        return engine

    def remove_camera(self, camera_id: str) -> None:
        self._engines.pop(camera_id, None)

    def reconfigure(self, cameras: Iterable[CameraConfig]) -> None:
        """Apply a config change, preserving state for cameras whose tuning is
        unchanged so a config reload cannot silently reset an open incident."""
        incoming = {camera.id: camera for camera in cameras}
        for camera_id in list(self._engines):
            if camera_id not in incoming:
                self.remove_camera(camera_id)
        for camera_id, camera in incoming.items():
            existing = self._engines.get(camera_id)
            if existing is None:
                self.add_camera(camera)
            elif existing.camera != camera:
                if existing.state is State.CONFIRMED:
                    # Keep the live incident; swap tuning in place.
                    existing.camera = camera
                else:
                    self.add_camera(camera)

    def get(self, camera_id: str) -> CameraIncidentEngine | None:
        return self._engines.get(camera_id)

    def observe(self, result: FrameResult) -> list[IncidentEvent]:
        engine = self._engines.get(result.camera_id)
        if engine is None:
            log.debug("dropping result for unknown camera %s", result.camera_id)
            return []
        events = engine.observe(result)
        metrics.INCIDENTS_ACTIVE.set(self.active_count())
        return events

    def active_count(self) -> int:
        return sum(1 for engine in self._engines.values() if engine.state is State.CONFIRMED)

    def status(self) -> list[dict]:
        return [engine.status() for engine in self._engines.values()]

    def find_incident(self, incident_id: str) -> CameraIncidentEngine | None:
        for engine in self._engines.values():
            if engine.incident is not None and engine.incident.id == incident_id:
                return engine
        return None
