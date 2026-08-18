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
