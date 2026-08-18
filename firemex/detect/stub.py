"""Detectors that need no model weights.

:class:`StubDetector` is a colour/luminance heuristic. It exists so the whole
pipeline -- ingest, confirmation, escalation, dashboard -- can be developed, tested
and demonstrated without a 2.5 GB PyTorch install, and so CI can run end to end.
It is *not* a production detector: it will fire on sunsets, hi-vis vests and
headlights, which is precisely the class of error the neural detector exists to fix.

:class:`ScriptedDetector` replays a fixed sequence and is what the tests use when
they need exact control over detector output.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from .base import FIRE, SMOKE, BBox, Detection

#: Heuristic works on a coarse cell grid rather than per-pixel connected
#: components, which keeps it dependency-free and fast enough to be irrelevant
#: next to real inference.
GRID = 24


def _grid_regions(mask: np.ndarray, min_cells: int) -> list[tuple[int, int, int, int, float]]:
    """Group adjacent True cells into regions.

    Returns ``(row0, col0, row1, col1, fill)`` per region, where ``fill`` is the
    mean of the underlying mask over the region's bounding box.
    """
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    regions: list[tuple[int, int, int, int, float]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if len(cells) < min_cells:
                continue
            r0 = min(cell[0] for cell in cells)
            r1 = max(cell[0] for cell in cells) + 1
            c0 = min(cell[1] for cell in cells)
            c1 = max(cell[1] for cell in cells) + 1
            fill = len(cells) / max((r1 - r0) * (c1 - c0), 1)
            regions.append((r0, c0, r1, c1, fill))
    regions.sort(key=lambda reg: (reg[2] - reg[0]) * (reg[3] - reg[1]), reverse=True)
    return regions


def _cell_means(channel: np.ndarray) -> np.ndarray:
    """Mean-pool an HxW plane down to a GRID x GRID grid."""
    h, w = channel.shape
    rows = np.array_split(np.arange(h), min(GRID, h))
    cols = np.array_split(np.arange(w), min(GRID, w))
    out = np.empty((len(rows), len(cols)), dtype=np.float32)
    for i, row_idx in enumerate(rows):
        band = channel[row_idx[0] : row_idx[-1] + 1, :]
        for j, col_idx in enumerate(cols):
            out[i, j] = float(band[:, col_idx[0] : col_idx[-1] + 1].mean())
    return out


class StubDetector:
    """Colour-heuristic fire/smoke detector. Development and CI use only."""

    name = "stub"

    def __init__(
        self,
        fire_confidence: float = 0.62,
        smoke_confidence: float = 0.55,
        min_cells: int = 3,
        max_detections: int = 4,
    ) -> None:
        self.fire_confidence = fire_confidence
        self.smoke_confidence = smoke_confidence
        self.min_cells = min_cells
        self.max_detections = max_detections

    def warmup(self) -> None:  # pragma: no cover - nothing to warm
        return None

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [self._predict_one(image) for image in images]

    def _predict_one(self, image: np.ndarray) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] < 3:
            return []
        rgb = image[:, :, :3].astype(np.float32)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # Fire: bright, strongly red-dominant.
        fire_mask = (red > 170) & (red > green * 1.28) & (green >= blue * 0.85)
        # Smoke: mid-bright and near-achromatic (low channel spread).
        brightness = rgb.mean(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        smoke_mask = (brightness > 85) & (brightness < 215) & (spread < 26)

        detections: list[Detection] = []
        detections += self._regions_to_detections(fire_mask, FIRE, self.fire_confidence, 0.35)
        detections += self._regions_to_detections(smoke_mask, SMOKE, self.smoke_confidence, 0.55)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]

    def _regions_to_detections(
        self, mask: np.ndarray, label: str, base_confidence: float, cell_fill: float
    ) -> list[Detection]:
        cells = _cell_means(mask.astype(np.float32))
        regions = _grid_regions(cells > cell_fill, self.min_cells)
        rows, cols = cells.shape
        out = []
        for r0, c0, r1, c1, fill in regions[: self.max_detections]:
            box = BBox(c0 / cols, r0 / rows, c1 / cols, r1 / rows).clipped()
            # Denser regions read as more confident, capped below certainty so the
            # heuristic can never present itself as more sure than a real model.
            confidence = min(0.94, base_confidence + 0.3 * (fill - cell_fill))
            out.append(Detection(label=label, confidence=confidence, box=box))
        return out


class ScriptedDetector:
    """Replays a caller-supplied sequence of per-frame detections.

    The sequence is consumed one entry per image. Once exhausted it repeats the
    ``loop`` value (empty by default), so a test can script a burst of detections
    followed by silence without padding the list.
    """

    name = "scripted"

    def __init__(
        self,
        script: Iterable[Sequence[Detection]],
        loop: Sequence[Detection] | None = None,
    ) -> None:
        self._script = deque(list(frame) for frame in script)
        self._loop = list(loop or [])
        self.calls = 0
        self.batch_sizes: list[int] = []

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        self.calls += 1
        self.batch_sizes.append(len(images))
        out = []
        for _ in images:
            out.append(list(self._script.popleft()) if self._script else list(self._loop))
        return out
