"""Channel-order conventions for the real inference backends.

These tests exist because of a bug that made the detector completely blind while
every other test passed: FiremeX carries RGB frames, but the Ultralytics
``predict()`` numpy path assumes BGR and flips internally. Handing it RGB swaps
red and blue, and because fire detection is overwhelmingly a colour cue, real
flames scored *nothing* rather than raising an error.

The stub and scripted detectors used elsewhere in the suite are colour-agnostic,
so nothing else can catch this. The two conventions are opposite and both are
pinned here:

* Ultralytics ``predict()`` -> **BGR** (it flips to RGB itself).
* ONNX exported graph      -> **RGB** (we build the tensor, post-flip layout).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from firemex.detect.onnx_backend import letterbox

# A frame that is unambiguously red-dominant, like flame. If a backend swaps
# channels, the blue plane ends up carrying the signal and the swap is visible.
FLAME_RGB = np.zeros((64, 96, 3), dtype=np.uint8)
FLAME_RGB[:, :, 0] = 240  # R
FLAME_RGB[:, :, 1] = 90   # G
FLAME_RGB[:, :, 2] = 20   # B


class _RecordingModel:
    """Stands in for ultralytics.YOLO and records what it was handed."""

    def __init__(self, *_args, **_kwargs):
        self.names = {0: "fire", 1: "other", 2: "smoke"}
        self.received: list[np.ndarray] = []

    def predict(self, images, **_kwargs):
        self.received = [np.asarray(image) for image in images]
        return []


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Install a fake ``ultralytics`` module so no weights or torch are needed."""
    module = types.ModuleType("ultralytics")
    module.YOLO = _RecordingModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


def test_ultralytics_backend_hands_the_model_bgr(fake_ultralytics):
    """The regression test for the blindness bug.

    Ultralytics flips numpy input BGR->RGB itself, so we must give it BGR. If this
    assertion fails, the real model scores nothing on real fire and no other test
    in the suite will notice.
    """
    from firemex.detect.ultralytics_backend import UltralyticsDetector

    detector = UltralyticsDetector("fake.pt", device="cpu", image_size=640)
    detector.predict([FLAME_RGB])

    handed = detector.model.received[0]
    assert handed.shape == FLAME_RGB.shape
    # Channel 0 must now be blue and channel 2 red -- i.e. reversed.
    assert handed[0, 0, 0] == 20, "channel 0 should be blue (BGR)"
    assert handed[0, 0, 1] == 90
    assert handed[0, 0, 2] == 240, "channel 2 should be red (BGR)"
    assert np.array_equal(handed, FLAME_RGB[:, :, ::-1])


def test_ultralytics_backend_does_not_mutate_the_caller_frame(fake_ultralytics):
    """The same frame goes into the ring buffer and the snapshot annotator."""
    from firemex.detect.ultralytics_backend import UltralyticsDetector

    original = FLAME_RGB.copy()
    detector = UltralyticsDetector("fake.pt", device="cpu")
    detector.predict([FLAME_RGB])
    assert np.array_equal(FLAME_RGB, original)


def test_ultralytics_backend_hands_over_a_contiguous_array(fake_ultralytics):
    """A reversed numpy view is non-contiguous; some torch paths reject it."""
    from firemex.detect.ultralytics_backend import UltralyticsDetector

    detector = UltralyticsDetector("fake.pt", device="cpu")
    detector.predict([FLAME_RGB])
    assert detector.model.received[0].flags["C_CONTIGUOUS"]


def test_ultralytics_backend_drops_the_other_class(fake_ultralytics):
    """The real checkpoint emits fire/other/smoke. 'other' must never escalate."""
    from firemex.detect.ultralytics_backend import UltralyticsDetector

    detector = UltralyticsDetector("fake.pt", device="cpu")
    assert detector._label_map == {0: "fire", 1: None, 2: "smoke"}


def test_onnx_preprocessing_preserves_rgb():
    """The mirror convention: we build this tensor, and the exported graph expects
    the post-flip RGB layout, so letterbox must not reorder channels."""
    canvas, _scale, pad_x, pad_y = letterbox(FLAME_RGB, 128)
    pixel = canvas[pad_y + 5, pad_x + 5]
    assert pixel[0] == 240, "channel 0 must stay red (RGB)"
    assert pixel[1] == 90
    assert pixel[2] == 20, "channel 2 must stay blue (RGB)"


def test_onnx_preprocessing_does_not_mutate_the_caller_frame():
    original = FLAME_RGB.copy()
    letterbox(FLAME_RGB, 128)
    assert np.array_equal(FLAME_RGB, original)


# ---- ONNX output layouts -------------------------------------------------
#
# Ultralytics emits two incompatible layouts and which one you get depends on the
# head. Assuming the raw layout made the ONNX backend return nothing at all for a
# real end-to-end export -- silent blindness again, so both are pinned here.


class _FakeMeta:
    def __init__(self, mapping):
        self.custom_metadata_map = mapping


class _FakeSession:
    """Minimal onnxruntime session stand-in returning a canned tensor."""

    def __init__(self, output, metadata=None):
        self._output = output
        self._metadata = _FakeMeta(metadata or {})

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def get_modelmeta(self):
        return self._metadata

    def run(self, _outputs, _feed):
        return [self._output]


def _detector(output, metadata=None, names=None):
    """Build an OnnxDetector around a fake session, bypassing __init__."""
    from firemex.detect.base import canonical_label
    from firemex.detect.onnx_backend import OnnxDetector

    detector = OnnxDetector.__new__(OnnxDetector)
    detector.session = _FakeSession(output, metadata)
    detector.input_name = "images"
    detector.image_size = 640
    detector.confidence_floor = 0.15
    detector.iou = 0.45
    detector._names = names or {0: "fire", 1: "other", 2: "smoke"}
    detector._label_map = {i: canonical_label(n) for i, n in detector._names.items()}
    detector._end2end = detector._end2end_from_metadata()
    return detector


def test_end2end_layout_is_decoded():
    """(n, 6) of [x1, y1, x2, y2, score, class] in letterbox pixels."""
    # A 640x640 input needs no letterbox padding, so pixels map straight through.
    rows = np.array(
        [
            [64.0, 128.0, 320.0, 448.0, 0.90, 0.0],   # fire
            [0.0, 0.0, 64.0, 64.0, 0.80, 1.0],        # "other" -> dropped
            [320.0, 320.0, 576.0, 576.0, 0.70, 2.0],  # smoke
            [0.0, 0.0, 10.0, 10.0, 0.01, 0.0],        # below the floor -> dropped
        ],
        dtype=np.float32,
    )
    detector = _detector(rows[None], metadata={"end2end": "True"})
    assert detector._end2end is True

    detections = detector.predict([np.zeros((640, 640, 3), dtype=np.uint8)])[0]
    assert [d.label for d in detections] == ["fire", "smoke"]
    assert detections[0].confidence == pytest.approx(0.90)
    # 64/640 = 0.1, 128/640 = 0.2, 320/640 = 0.5, 448/640 = 0.7
    assert detections[0].box.x1 == pytest.approx(0.1)
    assert detections[0].box.y1 == pytest.approx(0.2)
    assert detections[0].box.x2 == pytest.approx(0.5)
    assert detections[0].box.y2 == pytest.approx(0.7)


def test_end2end_layout_is_sniffed_when_metadata_is_silent():
    """Older exports omit the flag, so the shape has to be inspected."""
    rows = np.array([[10.0, 10.0, 100.0, 100.0, 0.75, 2.0]], dtype=np.float32)
    detector = _detector(rows[None], metadata={})
    assert detector._end2end is None, "metadata says nothing"
    detections = detector.predict([np.zeros((640, 640, 3), dtype=np.uint8)])[0]
    assert [d.label for d in detections] == ["smoke"]


def test_raw_layout_still_works_and_is_suppressed():
    """(4 + nc, n) of cxcywh plus per-class scores, needing NMS here."""
    # Two near-identical boxes plus a distant one; NMS should drop the duplicate.
    raw = np.zeros((7, 3), dtype=np.float32)
    for i, (cx, cy, w, h, fire) in enumerate(
        [(320, 320, 128, 128, 0.90), (322, 322, 128, 128, 0.85), (100, 100, 40, 40, 0.70)]
    ):
        raw[0, i], raw[1, i], raw[2, i], raw[3, i] = cx, cy, w, h
        raw[4, i] = fire  # class 0 = fire
    detector = _detector(raw[None], metadata={"end2end": "False"})
    assert detector._end2end is False

    detections = detector.predict([np.zeros((640, 640, 3), dtype=np.uint8)])[0]
    assert len(detections) == 2, f"NMS should have merged the duplicate: {detections}"
    assert all(d.label == "fire" for d in detections)


def test_a_raw_two_class_export_is_not_mistaken_for_end2end():
    """A 2-class raw export is also 6 columns wide, which is why metadata wins."""
    raw = np.zeros((6, 1), dtype=np.float32)
    raw[0, 0], raw[1, 0], raw[2, 0], raw[3, 0] = 320, 320, 64, 64
    raw[4, 0] = 0.9  # fire score
    detector = _detector(raw[None], metadata={"end2end": "False"}, names={0: "fire", 1: "smoke"})
    detections = detector.predict([np.zeros((640, 640, 3), dtype=np.uint8)])[0]
    assert [d.label for d in detections] == ["fire"]


def test_letterbox_uses_smooth_resampling():
    """Nearest-neighbour striding cost roughly half the confidence on smoke, a soft
    low-contrast texture. A smooth kernel must average, not pick, when downscaling."""
    # A 2px checkerboard averages to a flat mid grey under any real filter, but
    # survives as hard black/white under nearest-neighbour.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[::2, ::2] = 255
    image[1::2, 1::2] = 255
    canvas, _scale, pad_x, pad_y = letterbox(image, 16)
    patch = canvas[pad_y + 2 : pad_y + 6, pad_x + 2 : pad_x + 6, 0].astype(int)
    assert patch.min() > 40 and patch.max() < 215, f"looks unfiltered: {patch.tolist()}"
