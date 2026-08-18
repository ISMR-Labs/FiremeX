"""ONNX Runtime backend -- the recommended production path.

Export once at deploy time:

    yolo export model=weights/firemex-yolov26s.pt format=onnx imgsz=640 dynamic=True

Implements letterbox preprocessing and NMS directly so no torch import is needed
at runtime. Handles both YOLO output layouts: ``(1, 4+nc, n)`` from v8-style
exports and ``(1, n, 4+nc)``.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BBox, Detection, canonical_label

log = logging.getLogger(__name__)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio onto a square grey canvas.

    Returns the canvas plus the scale and padding needed to map boxes back.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    # Nearest-neighbour via index arithmetic keeps this dependency-free; the
    # detector is trained with heavy scale augmentation so interpolation quality
    # here is not a meaningful accuracy factor.
    rows = (np.arange(new_h) / scale).astype(np.int32).clip(0, height - 1)
    cols = (np.arange(new_w) / scale).astype(np.int32).clip(0, width - 1)
    resized = image[rows][:, cols]
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y, pad_x = (size - new_h) // 2, (size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized[:, :, :3]
    return canvas, scale, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression over xyxy boxes."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = areas[best] + areas[rest] - overlap
        ious = np.where(union > 0, overlap / union, 0.0)
        order = rest[ious <= threshold]
    return keep


class OnnxDetector:
    name = "onnx"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        image_size: int = 640,
        confidence_floor: float = 0.15,
        iou: float = 0.45,
        class_names: dict[int, str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The onnx backend needs the 'onnx' extra: pip install 'firemex[onnx]'"
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = image_size
        self.confidence_floor = confidence_floor
        self.iou = iou
        # Prefer names embedded by the Ultralytics exporter, else the caller's map,
        # else assume the conventional fire/smoke ordering.
        self._names = class_names or self._names_from_metadata() or {0: "fire", 1: "smoke"}
        self._label_map = {idx: canonical_label(name) for idx, name in self._names.items()}
        log.info("onnx providers=%s classes=%s", providers, self._names)

    def _names_from_metadata(self) -> dict[int, str] | None:
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        try:
            import ast

            parsed = ast.literal_eval(raw)
            return {int(k): str(v) for k, v in dict(parsed).items()}
        except (ValueError, SyntaxError, TypeError):  # pragma: no cover - odd export
            log.warning("could not parse class names from ONNX metadata")
            return None

    def warmup(self) -> None:
        self.predict([np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)])

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        batch = []
        transforms = []
        for image in images:
            canvas, scale, pad_x, pad_y = letterbox(image, self.image_size)
            batch.append(canvas.transpose(2, 0, 1).astype(np.float32) / 255.0)
            transforms.append((scale, pad_x, pad_y, image.shape[0], image.shape[1]))
        outputs = self.session.run(None, {self.input_name: np.stack(batch)})[0]
        return [
            self._decode(outputs[i], *transforms[i]) for i in range(min(len(images), len(outputs)))
        ]

    def _decode(
        self, raw: np.ndarray, scale: float, pad_x: int, pad_y: int, height: int, width: int
    ) -> list[Detection]:
        predictions = np.asarray(raw, dtype=np.float32)
        num_classes = len(self._names)
        # Normalise to (n, 4 + nc).
        if predictions.ndim == 2 and predictions.shape[0] == 4 + num_classes:
            predictions = predictions.transpose(1, 0)
        if predictions.ndim != 2 or predictions.shape[1] < 4 + num_classes:
            log.warning("unexpected ONNX output shape %s", raw.shape)
            return []

        xywh = predictions[:, :4]
        scores_by_class = predictions[:, 4 : 4 + num_classes]
        class_ids = scores_by_class.argmax(axis=1)
        scores = scores_by_class.max(axis=1)
        keep_mask = scores >= self.confidence_floor
        if not keep_mask.any():
            return []
        xywh, scores, class_ids = xywh[keep_mask], scores[keep_mask], class_ids[keep_mask]

        # cxcywh (letterboxed pixels) -> xyxy in the original frame.
        boxes = np.empty_like(xywh)
        boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2 - pad_x) / scale
        boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2 - pad_y) / scale
        boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2 - pad_x) / scale
        boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2 - pad_y) / scale

        out: list[Detection] = []
        for index in nms(boxes, scores, self.iou):
            label = self._label_map.get(int(class_ids[index]))
            if label is None:
                continue
            x1, y1, x2, y2 = boxes[index]
            box = BBox(x1 / width, y1 / height, x2 / width, y2 / height).clipped()
            if box.area <= 0:
                continue
            out.append(Detection(label=label, confidence=float(scores[index]), box=box))
        return out
