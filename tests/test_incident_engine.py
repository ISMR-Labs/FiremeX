"""Confirmation rules.

This is the module that decides whether a phone rings, so each rule gets a test
that names the real-world false positive it exists to reject.
"""

from __future__ import annotations

import pytest

from firemex.detect.base import BBox, Detection
from firemex.incident.engine import (
    CameraIncidentEngine,
    EventKind,
    IncidentEngine,
    Reason,
    Severity,
    State,
    largest_stable_cluster,
)


def fire(confidence=0.9, box=None):
    return Detection("fire", confidence, box or BBox(0.40, 0.40, 0.60, 0.60))


def smoke(confidence=0.8, box=None):
    return Detection("smoke", confidence, box or BBox(0.40, 0.40, 0.60, 0.60))


def growing(step: int, base: float = 0.18, rate: float = 0.004):
    """A box centred at (0.5, 0.5) growing steadily but not explosively.

    Slow enough to stay under the fast-growth escalation threshold, so tests that
    care about the opening severity are not perturbed by it.
    """
    half = (base + step * rate) / 2
    return BBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)


def feed(engine, next_result, frames, detections_for, is_night=False):
    """Push ``frames`` frames through the engine, collecting every event."""
    events = []
    for step in range(frames):
        events += engine.observe(next_result(detections_for(step), is_night=is_night))
    return events


# ---- happy path ----------------------------------------------------------


def test_sustained_growing_fire_confirms(camera, result_factory, counter_ids):
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    events = feed(engine, result_factory(), 8, lambda i: [fire(box=growing(i))])

    opened = [e for e in events if e.kind is EventKind.OPENED]
    assert len(opened) == 1, "a sustained growing fire must confirm exactly once"
    assert engine.state is State.CONFIRMED
    assert opened[0].incident.severity is Severity.CRITICAL
    assert opened[0].incident.labels == {"fire"}
    assert opened[0].incident.id == "cam-1-1"


def test_confirmation_needs_the_configured_frame_count(camera, result_factory, counter_ids):
    """Five hits with frames_required=6 must not confirm; the sixth must."""
    subject = camera(confirm={"frames_required": 6, "window": 10, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    next_result = result_factory()

    for step in range(5):
        assert engine.observe(next_result([fire(box=growing(step))])) == []
    assert engine.state is State.CANDIDATE
    assert engine.last_assessment.rejected is Reason.INSUFFICIENT_PERSISTENCE

    events = engine.observe(next_result([fire(box=growing(5))]))
    assert [e.kind for e in events] == [EventKind.OPENED]


def test_smoke_only_opens_as_warning(camera, result_factory, counter_ids):
    """Smoke first is how a real fire normally presents, so it must still alert."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    events = feed(engine, result_factory(), 8, lambda i: [smoke(box=growing(i))])
    opened = [e for e in events if e.kind is EventKind.OPENED]
    assert len(opened) == 1
    assert opened[0].severity is Severity.WARNING
    assert opened[0].labels == frozenset({"smoke"})


# ---- persistence: single-frame flukes ------------------------------------


def test_single_frame_fluke_is_rejected(camera, result_factory, counter_ids):
    """One frame of compression artefact must never ring a phone."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    next_result = result_factory()
    assert engine.observe(next_result([fire()])) == []
    for _ in range(10):
        assert engine.observe(next_result([])) == []
    assert engine.state is State.IDLE


def test_intermittent_detections_below_quorum_never_confirm(camera, result_factory, counter_ids):
    """Detection on every third frame: present, but not persistent."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    events = feed(
        engine, result_factory(), 30, lambda i: [fire(box=growing(i))] if i % 3 == 0 else []
    )
    assert not events
    assert engine.incident is None


# ---- spatial stability: the sweeping headlight ---------------------------


def test_moving_detection_is_rejected(camera, result_factory, counter_ids):
    """A detection that sweeps across the frame is a headlight or a hi-vis vest,
    not a fire. Persistent, but not in one place."""
    subject = camera(confirm={"frames_required": 6, "window": 10, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)

    def sweeping(step):
        x = 0.02 + step * 0.11
        return [fire(box=BBox(x, 0.4, x + 0.08, 0.5))]

    events = feed(engine, result_factory(), 8, sweeping)
    assert not events, "a sweeping detection must not confirm"
    assert engine.last_assessment.rejected is Reason.NOT_SPATIALLY_STABLE
    assert engine.last_assessment.hit_frames >= engine.last_assessment.frames_required


def test_stationary_detection_passes_stability(camera, result_factory, counter_ids):
    subject = camera(confirm={"frames_required": 6, "window": 10, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    events = feed(engine, result_factory(), 8, lambda i: [fire()])
    assert [e.kind for e in events] == [EventKind.OPENED]


def test_largest_stable_cluster_counts_overlapping_boxes():
    stationary = [BBox(0.4, 0.4, 0.6, 0.6) for _ in range(5)]
    assert largest_stable_cluster(stationary, 0.2) == 5

    scattered = [BBox(x, 0.4, x + 0.05, 0.45) for x in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert largest_stable_cluster(scattered, 0.2) == 1
    assert largest_stable_cluster([], 0.2) == 0


# ---- growth --------------------------------------------------------------


def test_collapsing_detection_is_rejected_as_flicker(camera, result_factory, counter_ids):
    subject = camera(
        confirm={
            "frames_required": 6,
            "window": 10,
            "require_growth": True,
            "growth_tolerance": 0.1,
        }
    )
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)

    def shrinking(step):
        half = max(0.02, (0.40 - step * 0.055)) / 2
        return [fire(box=BBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half))]

    events = feed(engine, result_factory(), 8, shrinking)
    assert not events
    assert engine.last_assessment.rejected is Reason.SHRINKING


def test_growth_check_can_be_disabled(camera, result_factory, counter_ids):
    subject = camera(confirm={"frames_required": 6, "window": 10, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)

    def shrinking(step):
        half = max(0.02, (0.40 - step * 0.055)) / 2
        return [fire(box=BBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half))]

    events = feed(engine, result_factory(), 8, shrinking)
    assert [e.kind for e in events] == [EventKind.OPENED]


def test_fast_growing_smoke_escalates_to_critical(camera, result_factory, counter_ids):
    """Smoke alone opens as a warning, but smoke behaving like a developing fire
    must escalate without waiting for visible flame."""
    subject = camera(
        confirm={
            "frames_required": 4,
            "window": 6,
            "require_growth": False,
            "growth_window_seconds": 120.0,
        }
    )
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    next_result = result_factory(fps=3.0)

    events = []
    for step in range(30):
        half = (0.06 + step * 0.012) / 2
        box = BBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)
        events += engine.observe(next_result([smoke(box=box)]))

    kinds = [e.kind for e in events]
    assert EventKind.OPENED in kinds
    assert EventKind.ESCALATED in kinds
    assert engine.incident.severity is Severity.CRITICAL
    assert engine.incident.growth_ratio > 1.6


# ---- filtering -----------------------------------------------------------


def test_below_threshold_detections_are_dropped(camera, result_factory, counter_ids):
    subject = camera()  # day fire threshold 0.40
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    events = feed(engine, result_factory(), 12, lambda i: [fire(confidence=0.30)])
    assert not events
    assert engine.state is State.IDLE


def test_night_thresholds_are_stricter(camera, result_factory, counter_ids):
    """0.45 clears the daytime fire floor (0.40) but not the night one (0.50).

    IR footage behaves differently from daylight, so this separation is what keeps
    a night-mode camera from alerting on its own illuminator.
    """
    subject = camera()
    day_engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    day_events = feed(
        day_engine, result_factory(), 8, lambda i: [fire(0.45, growing(i))], is_night=False
    )
    assert [e.kind for e in day_events] == [EventKind.OPENED]

    night_engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    night_events = feed(
        night_engine, result_factory(), 8, lambda i: [fire(0.45, growing(i))], is_night=True
    )
    assert not night_events


def test_tiny_boxes_are_dropped(camera, result_factory, counter_ids):
    subject = camera(confirm={"min_box_area": 0.01})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    tiny = BBox(0.50, 0.50, 0.52, 0.52)  # area 0.0004
    events = feed(engine, result_factory(), 12, lambda i: [fire(box=tiny)])
    assert not events


def test_detections_inside_exclusion_zone_are_dropped(camera, result_factory, counter_ids):
    """The sunset through the west window, excluded geometrically."""
    window = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)]
    subject = camera(exclude_zones=[window])
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)

    inside = BBox(0.05, 0.05, 0.30, 0.30)
    assert not feed(engine, result_factory(), 12, lambda i: [fire(box=inside)])

    # The same detection elsewhere in the frame must still confirm.
    elsewhere = CameraIncidentEngine(subject, id_factory=counter_ids)
    events = feed(elsewhere, result_factory(), 8, lambda i: [fire(box=growing(i))])
    assert [e.kind for e in events] == [EventKind.OPENED]


def test_unknown_labels_never_escalate(camera, result_factory, counter_ids):
    """A checkpoint that also emits 'person' must not turn people into fires."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    person = Detection("person", 0.99, BBox(0.4, 0.4, 0.6, 0.6))
    assert not feed(engine, result_factory(), 12, lambda i: [person])


# ---- lifecycle -----------------------------------------------------------


def test_incident_closes_after_the_clear_timeout(camera, result_factory, counter_ids):
    subject = camera(confirm={"clear_after_seconds": 2.0})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    next_result = result_factory(fps=3.0)

    for step in range(8):
        engine.observe(next_result([fire(box=growing(step))]))
    assert engine.state is State.CONFIRMED

    closed = []
    for _ in range(10):  # 10 frames at 3 fps = 3.3s of quiet
        closed += engine.observe(next_result([]))

    assert [e.kind for e in closed] == [EventKind.CLOSED]
    assert engine.incident is None
    assert engine.state is State.IDLE
    assert closed[0].incident.closed_wall is not None


def test_one_fire_produces_one_incident(camera, result_factory, counter_ids):
    """A ten-minute fire must not open a new incident on every frame."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    events = feed(engine, result_factory(), 120, lambda i: [fire(box=growing(min(i, 40)))])
    assert sum(1 for e in events if e.kind is EventKind.OPENED) == 1


def test_cancel_suppresses_reopening_until_the_camera_goes_quiet(
    camera, result_factory, counter_ids
):
    """An operator who dismisses a sunset must not be re-alerted three frames later."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    next_result = result_factory()
    for step in range(8):
        engine.observe(next_result([fire(box=growing(step))]))
    assert engine.state is State.CONFIRMED

    assert engine.cancel("operator").kind is EventKind.CLOSED
    assert engine.incident is None

    # Same detection continues: must stay silent.
    for step in range(20):
        assert engine.observe(next_result([fire(box=growing(step))])) == []

    # Camera goes quiet, clearing the suppression...
    for _ in range(engine.camera.confirm.window):
        engine.observe(next_result([]))
    # ...and a genuinely new event confirms again.
    reopened = feed(engine, next_result and result_factory(start=2000.0), 8,
                    lambda i: [fire(box=growing(i))])
    assert [e.kind for e in reopened] == [EventKind.OPENED]


def test_labels_accumulate_and_escalate_smoke_to_fire(camera, result_factory, counter_ids):
    subject = camera(confirm={"frames_required": 4, "window": 6, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    next_result = result_factory()

    events = []
    for _ in range(5):
        events += engine.observe(next_result([smoke()]))
    assert engine.incident.severity is Severity.WARNING

    events += engine.observe(next_result([smoke(), fire()]))
    assert engine.incident.severity is Severity.CRITICAL
    assert engine.incident.labels == {"fire", "smoke"}
    assert any(e.kind is EventKind.ESCALATED for e in events)


def test_status_reports_why_it_did_not_confirm(camera, result_factory, counter_ids):
    """"Why didn't it call?" must always have an answer."""
    engine = CameraIncidentEngine(camera(), id_factory=counter_ids)
    next_result = result_factory()
    for _ in range(2):
        engine.observe(next_result([fire()]))

    status = engine.status()
    assert status["state"] == "candidate"
    assert status["window_hits"] == 2
    assert status["assessment"]["rejected"] == "insufficient_persistence"
    assert status["incident_id"] is None


# ---- multi-camera manager ------------------------------------------------


def test_engine_routes_per_camera(camera, result_factory, counter_ids):
    cam_a = camera(id="cam-a")
    cam_b = camera(id="cam-b")
    manager = IncidentEngine([cam_a, cam_b], id_factory=counter_ids)

    next_a = result_factory(camera_id="cam-a")
    next_b = result_factory(camera_id="cam-b")
    events = []
    for step in range(8):
        events += manager.observe(next_a([fire(box=growing(step))]))
        events += manager.observe(next_b([]))

    assert len(events) == 1
    assert events[0].incident.camera_id == "cam-a"
    assert manager.active_count() == 1
    assert manager.get("cam-b").state is State.IDLE


def test_engine_ignores_unknown_cameras(camera, result_factory, counter_ids):
    manager = IncidentEngine([camera(id="cam-a")], id_factory=counter_ids)
    stray = result_factory(camera_id="ghost")
    assert manager.observe(stray([fire()])) == []


def test_reconfigure_preserves_an_open_incident(camera, result_factory, counter_ids):
    """A config reload must not silently drop a live incident."""
    subject = camera()
    manager = IncidentEngine([subject], id_factory=counter_ids)
    next_result = result_factory()
    for step in range(8):
        manager.observe(next_result([fire(box=growing(step))]))
    incident_id = manager.get("cam-1").incident.id

    retuned = camera(thresholds={"day": {"fire": 0.55, "smoke": 0.6}})
    manager.reconfigure([retuned])

    engine = manager.get("cam-1")
    assert engine.state is State.CONFIRMED
    assert engine.incident.id == incident_id
    assert engine.camera.thresholds.day.fire == pytest.approx(0.55)


def test_reconfigure_removes_deleted_cameras(camera, counter_ids):
    manager = IncidentEngine([camera(id="cam-a"), camera(id="cam-b")], id_factory=counter_ids)
    manager.reconfigure([camera(id="cam-a")])
    assert manager.get("cam-a") is not None
    assert manager.get("cam-b") is None


def test_find_incident_by_id(camera, result_factory, counter_ids):
    manager = IncidentEngine([camera()], id_factory=counter_ids)
    next_result = result_factory()
    for step in range(8):
        manager.observe(next_result([fire(box=growing(step))]))
    incident_id = manager.get("cam-1").incident.id
    assert manager.find_incident(incident_id) is manager.get("cam-1")
    assert manager.find_incident("nope") is None


def test_event_snapshots_severity_at_emission(camera, result_factory, counter_ids):
    """The event carries the live Incident, so anything inspecting an event after
    the fact must read the snapshot rather than the mutated object."""
    subject = camera(confirm={"frames_required": 4, "window": 6, "require_growth": False})
    engine = CameraIncidentEngine(subject, id_factory=counter_ids)
    next_result = result_factory()

    events = []
    for _ in range(5):
        events += engine.observe(next_result([smoke()]))
    opened = next(e for e in events if e.kind is EventKind.OPENED)
    assert opened.severity is Severity.WARNING

    # Flame appears and the incident escalates in place.
    engine.observe(next_result([smoke(), fire()]))
    assert engine.incident.severity is Severity.CRITICAL
    assert opened.severity is Severity.WARNING, "the opened event must not be rewritten"
    assert opened.incident.severity is Severity.CRITICAL, "the incident itself is live"
