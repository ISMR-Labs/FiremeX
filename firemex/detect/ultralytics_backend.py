"""Ultralytics YOLO backend (PyTorch).

Use this for development, fine-tuning and evaluation. For deployment prefer the
ONNX backend: it is 2-4x faster on the same hardware and drops the torch install.

Licensing note: Ultralytics YOLO is AGPL-3.0. If FiremeX is ever distributed as a
closed product, ship the ONNX runtime path instead of this module.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BBox, Detection, canonical_label

log = logging.getLogger(__name__)


class UltralyticsDetector:
    name = "ultralytics"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        image_size: int = 640,
        confidence_floor: float = 0.15,
        iou: float = 0.45,
        half: bool | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The ultralytics backend needs the 'torch' extra: pip install 'firemex[torch]'"
            ) from exc

        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        # Predict below the operating threshold and let the incident engine apply the
        # real per-class, day/night cut. Filtering twice would make the configured
        # thresholds unreachable.
        self.confidence_floor = confidence_floor
        self.iou = iou
        self.half = half if half is not None else device.startswith("cuda")
        self.model = YOLO(model_path)
        self._names: dict[int, str] = dict(self.model.names or {})
        self._label_map = {idx: canonical_label(name) for idx, name in self._names.items()}
        dropped = [self._names[i] for i, lab in self._label_map.items() if lab is None]
        if dropped:
            log.info("ignoring non-fire classes from checkpoint: %s", ", ".join(sorted(dropped)))

    def warmup(self) -> None:
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.predict([blank])

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        results = self.model.predict(
            images,
            imgsz=self.image_size,
            conf=self.confidence_floor,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        return [self._convert(result) for result in results]

    def _convert(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        height, width = result.orig_shape[:2]
        out: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, classes, strict=True):
            label = self._label_map.get(int(class_id))
            if label is None:
                continue
            box = BBox(
                float(x1) / width, float(y1) / height, float(x2) / width, float(y2) / height
            ).clipped()
            out.append(Detection(label=label, confidence=float(confidence), box=box))
        return out
