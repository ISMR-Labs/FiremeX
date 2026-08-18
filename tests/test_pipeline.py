"""End-to-end pipeline.

Camera -> inference -> confirmation -> evidence -> escalation, driven through the
real Supervisor with a synthetic camera and the stub detector. No hardware, no
model download, no Twilio account. This is the test that catches wiring mistakes
the unit tests cannot see.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from firemex.config import CameraConfig, ContactConfig, SiteConfig
from firemex.detect.base import BBox, Detection
from firemex.detect.stub import ScriptedDetector, StubDetector
from firemex.ingest.recorder import annotate, write_snapshot
from firemex.ingest.ringbuffer import FrameRingBuffer
from firemex.ingest.sources import SyntheticSource
from firemex.notify.base import AlertContext, AlertResult, Outcome
from firemex.supervisor import Supervisor


class CapturingChannel:
    """Stands in for Twilio and records what would have been sent."""

    def __init__(self, name="call"):
        self.name = name
        self.calls: list[str] = []

    async def send(self, context: AlertContext, contact, message: str) -> AlertResult:
        self.calls.append(contact.id)
        return AlertResult(self.name, contact.id, Outcome.QUEUED, provider_id="sid-test")


def build_site(**alerting) -> SiteConfig:
    return SiteConfig(
        name="Pipeline Test Site",
        timezone="UTC",
        cameras=[
            CameraConfig(
                id="sim-1",
                name="Simulated Bay",
                location="Test",
                rtsp="synthetic://sim-1",
                sample_fps=10.0,
                confirm={
                    "frames_required": 4,
                    "window": 6,
                    "require_growth": False,
                    "clear_after_seconds": 1.0,
                },
            )
        ],
        contacts=[
            ContactConfig(
                id="primary",
                name="Primary",
                phone="+10000000001",
                channels=["call"],
                retries=0,
                escalate_after_seconds=0.05,
            )
        ],
        alerting={"default_contacts": ["primary"], **alerting},
    )


async def wait_for(predicate, timeout=8.0, interval=0.05):
    """Poll until ``predicate`` is true, so the test never sleeps longer than needed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ---- the full path -------------------------------------------------------


async def test_synthetic_fire_travels_the_whole_pipeline(settings):
    """A growing fire on a synthetic camera must confirm, record evidence, and
    reach the escalation chain."""
    settings.shadow_mode = False
    channel = CapturingChannel()
    site = build_site(confirm_delay_seconds=0.0, cooldown_minutes=10)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=StubDetector(),
        source_factory=lambda camera: SyntheticSource(
            width=320, height=180, fps=15.0, ignite_after=0.2, ramp_seconds=3.0
        ),
        channels={"call": channel},
    )
    await supervisor.start()
    try:
        assert await wait_for(lambda: supervisor.engine.active_count() == 1), (
            "synthetic fire never confirmed"
        )
        engine = supervisor.engine.get("sim-1")
        incident = engine.incident
        assert incident is not None
        assert "fire" in incident.labels

        # It reached the escalation chain.
        assert await wait_for(lambda: channel.calls == ["primary"]), (
            f"contact was never called (calls={channel.calls})"
        )

        # It was persisted with evidence.
        assert await wait_for(lambda: bool(incident.snapshot_path))
        stored = await supervisor.store.get_incident(incident.id)
        assert stored is not None
        assert stored["camera_name"] == "Simulated Bay"
        assert stored["shadow_mode"] is False
        assert stored["has_snapshot"] is True
        assert stored["detections"], "the confirming detections must be recorded"
    finally:
        await supervisor.stop()


async def test_shadow_mode_records_but_never_calls(settings):
    """The default posture for a new site: full audit trail, zero phone calls."""
    settings.shadow_mode = True
    channel = CapturingChannel()
    site = build_site(confirm_delay_seconds=0.0)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=StubDetector(),
        source_factory=lambda camera: SyntheticSource(
            width=320, height=180, fps=15.0, ignite_after=0.2, ramp_seconds=3.0
        ),
        channels={"call": channel},
    )
    await supervisor.start()
    try:
        assert await wait_for(lambda: supervisor.engine.active_count() == 1)
        incident_id = supervisor.engine.get("sim-1").incident.id

        # Poll the persisted outcome rather than the in-memory run: the run marks
        # itself finished *before* the store write it triggers has landed.
        stored = None

        async def outcome_recorded() -> bool:
            nonlocal stored
            stored = await supervisor.store.get_incident(incident_id)
            return bool(stored and stored["alert_status"])

        recorded = False
        for _ in range(160):
            if await outcome_recorded():
                recorded = True
                break
            await asyncio.sleep(0.05)
        assert recorded, "the shadow-mode outcome was never persisted"

        assert channel.calls == [], "shadow mode must never place a call"
        assert stored["shadow_mode"] is True
        assert stored["alert_status"] == "shadow"
    finally:
        await supervisor.stop()


async def test_operator_cancel_stops_the_calls_and_records_the_verdict(settings):
    settings.shadow_mode = False
    channel = CapturingChannel()
    # A long grace period so the cancel definitely lands inside it.
    site = build_site(confirm_delay_seconds=30.0)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=StubDetector(),
        source_factory=lambda camera: SyntheticSource(
            width=320, height=180, fps=15.0, ignite_after=0.2, ramp_seconds=3.0
        ),
        channels={"call": channel},
    )
    await supervisor.start()
    try:
        assert await wait_for(lambda: supervisor.engine.active_count() == 1)
        incident_id = supervisor.engine.get("sim-1").incident.id

        assert await supervisor.cancel_incident(incident_id, "operator: no fire") is True
        await asyncio.sleep(0.1)

        assert channel.calls == [], "cancelling inside the grace period must place no call"
        stored = await supervisor.store.get_incident(incident_id)
        assert stored["review"] == "false_positive"
        assert supervisor.engine.get("sim-1").incident is None
    finally:
        await supervisor.stop()


async def test_quiet_camera_produces_no_incident(settings):
    """A dark, empty scene must stay silent. The pipeline's resting state is quiet."""
    settings.shadow_mode = False
    channel = CapturingChannel()
    site = build_site(confirm_delay_seconds=0.0)

    class DarkSource:
        def open(self):
            return None

        def read(self):
            time.sleep(0.02)
            return np.full((180, 320, 3), 10, dtype=np.uint8)

        def close(self):
            return None

    supervisor = Supervisor(
        settings,
        site=site,
        detector=StubDetector(),
        source_factory=lambda camera: DarkSource(),
        channels={"call": channel},
    )
    await supervisor.start()
    try:
        await asyncio.sleep(1.0)
        assert supervisor.engine.active_count() == 0
        assert channel.calls == []
        assert await supervisor.store.list_incidents() == []
        # But the camera is genuinely running, not silently dead.
        assert supervisor.workers["sim-1"].frames_decoded > 0
    finally:
        await supervisor.stop()


async def test_status_is_reportable_while_running(settings):
    settings.shadow_mode = True
    site = build_site()
    supervisor = Supervisor(
        settings,
        site=site,
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: _IdleSource(),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        status = supervisor.status()
        assert status["site"] == "Pipeline Test Site"
        assert status["shadow_mode"] is True
        assert len(status["cameras"]) == 1
        assert status["cameras"][0]["connected"] is True
        assert status["contacts"][0]["id"] == "primary"
    finally:
        await supervisor.stop()


class _IdleSource:
    def open(self):
        return None

    def read(self):
        time.sleep(0.02)
        return None

    def close(self):
        return None


# ---- worker resilience ---------------------------------------------------


async def test_worker_reconnects_after_a_stream_failure(settings):
    """Cameras drop. The worker must keep reopening rather than quietly giving up."""
    from firemex.ingest.sources import StreamError

    attempts = {"count": 0}

    class FlakySource:
        def open(self):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise StreamError("connection refused")

        def read(self):
            time.sleep(0.02)
            return np.full((180, 320, 3), 10, dtype=np.uint8)

        def close(self):
            return None

    supervisor = Supervisor(
        settings,
        site=build_site(),
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: FlakySource(),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        worker = supervisor.workers["sim-1"]
        # Wait on a decoded frame, not on `connected`: the worker marks itself
        # connected the moment the stream opens, which is before any frame has
        # arrived.
        assert await wait_for(lambda: worker.frames_decoded > 0, timeout=10.0), (
            f"never recovered after {attempts['count']} attempts "
            f"(connected={worker.connected}, last_error={worker.last_error})"
        )
        assert worker.connected is True
        assert worker.reconnects >= 2, "both failed opens should be counted"
    finally:
        await supervisor.stop()


async def test_worker_drops_frames_rather_than_queueing_them(settings):
    """Under load a fire detector must stay current; a backlog would only convert
    overload into alert latency."""

    class BlockingDetector:
        name = "blocking"

        def warmup(self):
            return None

        def close(self):
            return None

        def predict(self, images):
            time.sleep(0.4)
            return [[] for _ in images]

    class FastSource:
        def open(self):
            return None

        def read(self):
            time.sleep(0.005)
            return np.full((180, 320, 3), 10, dtype=np.uint8)

        def close(self):
            return None

    site = build_site()
    site.cameras[0].sample_fps = 30.0
    supervisor = Supervisor(
        settings,
        site=site,
        detector=BlockingDetector(),
        source_factory=lambda camera: FastSource(),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        worker = supervisor.workers["sim-1"]
        assert await wait_for(lambda: worker.frames_dropped > 0, timeout=6.0), (
            "frames should have been dropped under a saturated detector"
        )
    finally:
        await supervisor.stop()


# ---- config reload -------------------------------------------------------


async def test_reload_applies_a_new_camera(settings, tmp_path):
    from firemex.config import dump_site_config

    site = build_site()
    dump_site_config(site, settings.config_path)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: _IdleSource(),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        assert set(supervisor.workers) == {"sim-1"}

        site.cameras.append(
            CameraConfig(id="sim-2", name="Second", rtsp="synthetic://sim-2", contacts=["primary"])
        )
        dump_site_config(site, settings.config_path)
        reloaded = await supervisor.reload_config()

        assert {c.id for c in reloaded.cameras} == {"sim-1", "sim-2"}
        assert set(supervisor.workers) == {"sim-1", "sim-2"}
        assert supervisor.engine.get("sim-2") is not None
    finally:
        await supervisor.stop()


async def test_reload_stops_a_removed_camera(settings):
    from firemex.config import dump_site_config

    site = build_site()
    site.cameras.append(
        CameraConfig(id="sim-2", name="Second", rtsp="synthetic://sim-2", contacts=["primary"])
    )
    dump_site_config(site, settings.config_path)

    supervisor = Supervisor(
        settings,
        site=site,
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: _IdleSource(),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        assert set(supervisor.workers) == {"sim-1", "sim-2"}
        site.cameras = [site.cameras[0]]
        dump_site_config(site, settings.config_path)
        await supervisor.reload_config()
        assert set(supervisor.workers) == {"sim-1"}
    finally:
        await supervisor.stop()


# ---- evidence ------------------------------------------------------------


def test_annotate_draws_without_mutating_the_source_frame():
    image = np.full((180, 320, 3), 40, dtype=np.uint8)
    original = image.copy()
    detections = [Detection("fire", 0.91, BBox(0.3, 0.3, 0.7, 0.7))]

    annotated = annotate(image, detections, caption="Camera 1 - Fire")

    assert annotated.shape == image.shape
    assert np.array_equal(image, original), "annotate must not modify its input"
    assert not np.array_equal(annotated, original), "annotate must draw something"


def test_snapshot_is_written_as_a_readable_jpeg(tmp_path):
    from PIL import Image

    path = tmp_path / "snap.jpg"
    image = np.full((180, 320, 3), 60, dtype=np.uint8)
    write_snapshot(path, image, [Detection("smoke", 0.7, BBox(0.1, 0.1, 0.5, 0.5))], "caption")

    assert path.exists()
    with Image.open(path) as opened:
        assert opened.size == (320, 180)
        assert opened.format == "JPEG"


def test_snapshot_creates_missing_directories(tmp_path):
    path = tmp_path / "nested" / "dirs" / "snap.jpg"
    write_snapshot(path, np.zeros((32, 32, 3), dtype=np.uint8))
    assert path.exists()


# ---- the pre-event ring buffer ------------------------------------------


def test_ring_buffer_evicts_by_age():
    """An incident clip is only useful if it shows what happened before the alarm."""
    buffer = FrameRingBuffer(seconds=1.0, max_frames=100)
    for step in range(30):
        buffer.push(step * 0.1, np.zeros((4, 4, 3), dtype=np.uint8))
    # 3 s of frames at 10 fps, keeping 1 s.
    assert len(buffer) <= 11
    assert buffer.snapshot()[0][0] >= 1.9


def test_ring_buffer_honours_the_frame_cap():
    """The duration cap alone would be unbounded memory on a high-fps stream."""
    buffer = FrameRingBuffer(seconds=1000.0, max_frames=5)
    for step in range(50):
        buffer.push(float(step), np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(buffer) == 5


def test_ring_buffer_since_returns_the_tail():
    buffer = FrameRingBuffer(seconds=100.0)
    for step in range(10):
        buffer.push(float(step), np.full((2, 2, 3), step, dtype=np.uint8))
    assert len(buffer.since(7.0)) == 3


def test_ring_buffer_latest_and_clear():
    buffer = FrameRingBuffer()
    assert buffer.latest() is None
    buffer.push(1.0, np.full((2, 2, 3), 9, dtype=np.uint8))
    assert buffer.latest()[0, 0, 0] == 9
    buffer.clear()
    assert buffer.latest() is None


# ---- the synthetic source ----------------------------------------------


def test_synthetic_source_is_dark_before_ignition():
    source = SyntheticSource(width=160, height=90, fps=1000.0, ignite_after=10.0)
    source.open()
    frame = source.read()
    assert frame.shape == (90, 160, 3)
    assert frame.max() < 100, "nothing should be burning yet"


def test_synthetic_source_produces_fire_colours_after_ignition():
    source = SyntheticSource(width=160, height=90, fps=1000.0, ignite_after=0.0, ramp_seconds=1.0)
    source.open()
    for _ in range(5):
        frame = source.read()
    red, green = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int)
    assert (red > 200).any(), "expected fire-coloured pixels"
    assert ((red > 170) & (red > green * 1.28)).any()


def test_synthetic_fire_grows():
    source = SyntheticSource(width=200, height=200, fps=1000.0, ignite_after=0.0, ramp_seconds=0.5)
    source.open()
    first = source.read()
    for _ in range(20):
        last = source.read()
    assert (last[:, :, 0] > 200).sum() > (first[:, :, 0] > 200).sum()


async def test_status_lists_open_incidents_for_a_reloaded_dashboard(settings):
    """A dashboard opened during a fire has no incident.opened event to replay, so
    status must carry the open incidents or the cancel button never appears."""
    settings.shadow_mode = True
    supervisor = Supervisor(
        settings,
        site=build_site(confirm_delay_seconds=0.0),
        detector=StubDetector(),
        source_factory=lambda camera: SyntheticSource(
            width=320, height=180, fps=15.0, ignite_after=0.2, ramp_seconds=3.0
        ),
        channels={"call": CapturingChannel()},
    )
    await supervisor.start()
    try:
        assert await wait_for(lambda: supervisor.engine.active_count() == 1)
        incident_id = supervisor.engine.get("sim-1").incident.id

        status = supervisor.status()
        assert status["active_incidents"] == 1
        listed = status["incidents"]
        assert [i["incident_id"] for i in listed] == [incident_id]
        # The card needs all of these to render.
        assert listed[0]["camera_name"] == "Simulated Bay"
        assert listed[0]["severity"] in {"warning", "critical"}
        assert listed[0]["labels"]
        assert listed[0]["shadow_mode"] is True

        # Once closed, status must stop listing it.
        await supervisor.cancel_incident(incident_id, "test")
        assert supervisor.status()["incidents"] == []
    finally:
        await supervisor.stop()
