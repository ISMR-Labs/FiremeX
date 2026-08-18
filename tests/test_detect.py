"""Detector plumbing: label normalisation, geometry, batching, ONNX decode maths."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from firemex.detect.base import FIRE, SMOKE, BBox, Detection, Frame, canonical_label, iou
from firemex.detect.onnx_backend import letterbox, nms
from firemex.detect.service import InferenceService
from firemex.detect.stub import ScriptedDetector, StubDetector

# ---- label normalisation -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fire", FIRE),
        ("Fire", FIRE),
        ("flame", FIRE),
        ("Active flames", FIRE),
        ("fire_indicator", FIRE),
        ("smoke", SMOKE),
        ("Smoke Plumes", SMOKE),
        ("smoke-plume", SMOKE),
    ],
)
def test_known_labels_normalise(raw, expected):
    assert canonical_label(raw) == expected


@pytest.mark.parametrize("raw", ["person", "car", "cloud", "dog", ""])
def test_unknown_labels_are_dropped_not_guessed(raw):
    """A checkpoint that also emits 'person' must not have it escalated to a fire."""
    assert canonical_label(raw) is None


# ---- geometry ------------------------------------------------------------


def test_iou_identical_boxes():
    box = BBox(0.2, 0.2, 0.4, 0.4)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou(BBox(0.0, 0.0, 0.1, 0.1), BBox(0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_half_overlap():
    a = BBox(0.0, 0.0, 0.2, 0.1)
    b = BBox(0.1, 0.0, 0.3, 0.1)
    # Intersection 0.1x0.1, union 0.03 -> 1/3.
    assert iou(a, b) == pytest.approx(1 / 3)


def test_bbox_properties():
    box = BBox(0.2, 0.1, 0.6, 0.5)
    assert box.width == pytest.approx(0.4)
    assert box.area == pytest.approx(0.16)
    assert box.centroid == pytest.approx((0.4, 0.3))
    assert box.to_pixels(1000, 500) == (200, 50, 600, 250)


def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValueError, match="degenerate"):
        BBox(0.6, 0.1, 0.2, 0.5)


def test_bbox_clipping_and_union():
    assert BBox(-0.2, -0.1, 1.4, 1.2).clipped() == BBox(0.0, 0.0, 1.0, 1.0)
    union = BBox(0.1, 0.1, 0.3, 0.3).union(BBox(0.5, 0.4, 0.7, 0.9))
    assert union == BBox(0.1, 0.1, 0.7, 0.9)


def test_detection_serialises_roundly():
    payload = Detection("fire", 0.87654, BBox(0.1, 0.2, 0.3, 0.4)).as_dict()
    assert payload["label"] == "fire"
    assert payload["confidence"] == pytest.approx(0.8765)
    assert payload["box"] == [0.1, 0.2, 0.3, 0.4]


# ---- the stub heuristic --------------------------------------------------


def test_stub_detects_a_fire_coloured_blob():
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    image[80:180, 120:230] = (250, 120, 30)
    detections = StubDetector().predict([image])[0]
    assert any(d.label == FIRE for d in detections)


def test_stub_finds_nothing_in_a_dark_frame():
    image = np.full((240, 320, 3), 12, dtype=np.uint8)
    assert StubDetector().predict([image])[0] == []


def test_stub_detects_grey_smoke():
    image = np.full((240, 320, 3), 20, dtype=np.uint8)
    image[40:160, 60:240] = (150, 150, 151)
    detections = StubDetector().predict([image])[0]
    assert any(d.label == SMOKE for d in detections)


def test_stub_returns_one_list_per_image():
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    assert len(StubDetector().predict(images)) == 5


def test_stub_never_claims_certainty():
    """The heuristic must not be able to present itself as more sure than a model."""
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    image[:, :] = (255, 130, 20)
    for detection in StubDetector().predict([image])[0]:
        assert detection.confidence <= 0.94


# ---- ONNX preprocessing / postprocessing --------------------------------


def test_letterbox_preserves_aspect_ratio_on_a_square_canvas():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert pad_x == 0
    assert pad_y == 140  # (640 - 360) // 2


def test_letterbox_scales_a_large_frame_down():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 1920)
    assert pad_x == 0
    assert pad_y == (640 - 360) // 2


def test_letterbox_pads_with_neutral_grey():
    image = np.full((100, 400, 3), 200, dtype=np.uint8)
    canvas, _, _, pad_y = letterbox(image, 400)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[pad_y + 10, 200].tolist() == [200, 200, 200]


def test_nms_suppresses_duplicate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep == [0, 2]


def test_nms_keeps_everything_when_nothing_overlaps():
    boxes = np.array([[0, 0, 5, 5], [20, 20, 25, 25], [40, 40, 45, 45]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1, 2]


def test_nms_returns_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float32)
    scores = np.array([0.3, 0.95], dtype=np.float32)
    assert nms(boxes, scores, 0.5)[0] == 1


# ---- the batched inference service ---------------------------------------


def make_frame(camera_id="cam-1", sequence=0):
    return Frame(
        camera_id=camera_id,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        monotonic_ts=float(sequence),
        wall_ts=1_700_000_000.0 + sequence,
        sequence=sequence,
    )


async def test_service_returns_a_result_per_frame():
    detector = ScriptedDetector(script=[[Detection("fire", 0.9, BBox(0.1, 0.1, 0.2, 0.2))]])
    service = InferenceService(detector, batch_size=4, batch_timeout_ms=5)
    await service.start()
    try:
        result = await service.submit(make_frame())
        assert result.camera_id == "cam-1"
        assert len(result.detections) == 1
        assert result.detections[0].label == "fire"
    finally:
        await service.stop()


async def test_service_batches_concurrent_submissions():
    """Batching across cameras is where multi-camera throughput comes from, so a
    burst of frames must reach the detector as one call, not eight."""
    detector = ScriptedDetector(script=[], loop=[])
    service = InferenceService(detector, batch_size=8, batch_timeout_ms=50)
    await service.start()
    try:
        frames = [make_frame(f"cam-{i}", i) for i in range(8)]
        results = await asyncio.gather(*(service.submit(frame) for frame in frames))
        assert len(results) == 8
        assert max(detector.batch_sizes) > 1, f"never batched: {detector.batch_sizes}"
        assert detector.calls < 8
    finally:
        await service.stop()


async def test_service_preserves_frame_identity_across_a_batch():
    detector = ScriptedDetector(script=[], loop=[])
    service = InferenceService(detector, batch_size=8, batch_timeout_ms=30)
    await service.start()
    try:
        frames = [make_frame(f"cam-{i}", sequence=i) for i in range(6)]
        results = await asyncio.gather(*(service.submit(frame) for frame in frames))
        assert [r.camera_id for r in results] == [f"cam-{i}" for i in range(6)]
        assert [r.sequence for r in results] == list(range(6))
    finally:
        await service.stop()


async def test_service_rejects_frames_when_saturated():
    """Backpressure must surface immediately so the caller can drop the frame; a
    growing queue would just convert overload into alert latency."""

    class SlowDetector:
        name = "slow"

        def warmup(self):
            return None

        def close(self):
            return None

        def predict(self, images):
            import time

            time.sleep(0.3)
            return [[] for _ in images]

    service = InferenceService(SlowDetector(), batch_size=1, batch_timeout_ms=1, queue_size=1)
    await service.start()
    try:
        # Frame 1 is picked up by the collector, which then blocks in the executor.
        in_flight = asyncio.create_task(service.submit(make_frame(sequence=1)))
        await asyncio.sleep(0.05)
        assert service.pending == 0

        # Frame 2 fills the now-idle queue slot; the collector cannot take it yet.
        queued = asyncio.create_task(service.submit(make_frame(sequence=2)))
        await asyncio.sleep(0.01)
        assert service.pending == 1

        # Frame 3 has nowhere to go and must be refused rather than buffered.
        with pytest.raises(asyncio.QueueFull):
            await service.submit(make_frame(sequence=3))
        assert service.dropped == 1

        for task in (in_flight, queued):
            task.cancel()
        await asyncio.gather(in_flight, queued, return_exceptions=True)
    finally:
        await service.stop()


async def test_service_survives_a_detector_exception():
    """One bad batch must not kill the inference loop for every camera."""

    class Flaky:
        name = "flaky"
        calls = 0

        def warmup(self):
            return None

        def close(self):
            return None

        def predict(self, images):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("CUDA hiccup")
            return [[] for _ in images]

    service = InferenceService(Flaky(), batch_size=1, batch_timeout_ms=1)
    await service.start()
    try:
        with pytest.raises(RuntimeError, match="CUDA hiccup"):
            await service.submit(make_frame())
        # The loop must still be alive and serving.
        result = await service.submit(make_frame(sequence=1))
        assert result.detections == []
    finally:
        await service.stop()


async def test_scripted_detector_falls_back_to_its_loop_value():
    detector = ScriptedDetector(script=[[Detection("fire", 0.9, BBox(0, 0, 0.1, 0.1))]], loop=[])
    assert len(detector.predict([np.zeros((8, 8, 3), dtype=np.uint8)])[0]) == 1
    assert detector.predict([np.zeros((8, 8, 3), dtype=np.uint8)])[0] == []
