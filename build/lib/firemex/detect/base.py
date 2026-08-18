"""Detector contract and the geometry primitives shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

#: Class labels FiremeX reasons about. Public fire/smoke checkpoints use a variety
#: of raw names ("Fire", "flame", "fire_indicator", ...); backends normalise onto these.
FIRE = "fire"
SMOKE = "smoke"
LABELS = (FIRE, SMOKE)

#: Raw checkpoint label -> canonical label. Extend when adopting new weights.
LABEL_ALIASES: dict[str, str] = {
    "fire": FIRE,
    "fires": FIRE,
    "flame": FIRE,
    "flames": FIRE,
    "active flames": FIRE,
    "active_flames": FIRE,
    "fire_indicator": FIRE,
    "fire indicators": FIRE,
    "smoke": SMOKE,
    "smokes": SMOKE,
    "smoke plume": SMOKE,
    "smoke_plume": SMOKE,
    "smoke plumes": SMOKE,
    "default": SMOKE,
}


def canonical_label(raw: str) -> str | None:
    """Map a checkpoint's class name onto ``fire``/``smoke``, or ``None`` to drop it.

    Unknown classes are dropped rather than guessed: a checkpoint that also emits
    "person" or "cloud" must not have those escalated into a fire alert.
    """
    key = raw.strip().lower().replace("-", " ")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for token, label in (("fire", FIRE), ("flame", FIRE), ("smoke", SMOKE)):
        if token in key:
            return label
    return None


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in normalised frame coordinates (0..1, origin top-left).

    Normalised rather than pixel coordinates so thresholds, exclusion zones and
    minimum-area rules survive a camera resolution change.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"degenerate box: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clipped(self) -> BBox:
        return BBox(
            min(max(self.x1, 0.0), 1.0),
            min(max(self.y1, 0.0), 1.0),
            min(max(self.x2, 0.0), 1.0),
            min(max(self.y2, 0.0), 1.0),
        )

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. 0.0 when the boxes do not overlap."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    denominator = a.area + b.area - intersection
    return intersection / denominator if denominator > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: BBox

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [
                round(self.box.x1, 5),
                round(self.box.y1, 5),
                round(self.box.x2, 5),
                round(self.box.y2, 5),
            ],
        }


@dataclass(slots=True)
class Frame:
    """One sampled frame on its way to the detector.

    ``image`` is HWC uint8 RGB. ``monotonic_ts`` drives all pipeline timing; wall
    clock is only for display and storage, so an NTP step cannot corrupt the
    confirmation logic.
    """

    camera_id: str
    image: np.ndarray
    monotonic_ts: float
    wall_ts: float
    is_night: bool = False
    sequence: int = 0

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class FrameResult:
    """Detector output for a single frame, before any temporal reasoning."""

    camera_id: str
    monotonic_ts: float
    wall_ts: float
    detections: list[Detection] = field(default_factory=list)
    is_night: bool = False
    inference_ms: float = 0.0
    sequence: int = 0


@runtime_checkable
class Detector(Protocol):
    """A batched fire/smoke detector.

    Implementations must be safe to call from a worker thread and must return one
    detection list per input image, in the same order.
    """

    name: str

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...
