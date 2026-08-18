"""ONNX Runtime backend -- the recommended production path.

Export once at deploy time:

    yolo export model=weights/firemex-yolov26s.pt format=onnx imgsz=640 dynamic=True

Implements letterbox preprocessing so no torch import is needed at runtime, and
handles the two very different output layouts Ultralytics produces:

**Raw** (v8-style, ``nms=False``) -- ``(1, 4+nc, n)`` or ``(1, n, 4+nc)`` of
``cxcywh`` boxes plus per-class scores, still needing NMS. Decoded and suppressed
here.

**End-to-end** (v10/v26-style, the default for newer heads) -- ``(1, n, 6)`` of
``[x1, y1, x2, y2, score, class_id]``, already suppressed inside the graph and
padded to a fixed row count. Nothing to do but filter and rescale.

Which one you get is recorded as ``end2end`` in the model metadata. Guessing from
shape alone is ambiguous -- a 2-class raw export is also 6 columns wide -- so the
metadata is trusted first and the shape only used as a fallback.
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
    from PIL import Image

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    # Bilinear via Pillow, which is already a hard dependency. Nearest-neighbour
    # index striding was measurably worse: it cost roughly half the confidence on
    # smoke, which is a soft low-contrast texture that aliases badly. Flame, being
    # high-contrast, barely noticed -- so this is exactly the kind of regression
    # that hides until you check the class you care about earliest.
    resized = np.asarray(
        Image.fromarray(np.ascontiguousarray(image[:, :, :3])).resize(
            (new_w, new_h), Image.BILINEAR
        )
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y, pad_x = (size - new_h) // 2, (size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
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
        self._end2end = self._end2end_from_metadata()
        log.info(
            "onnx providers=%s classes=%s end2end=%s",
            providers,
            self._names,
            self._end2end if self._end2end is not None else "auto",
        )

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

    def _end2end_from_metadata(self) -> bool | None:
        """Whether NMS is baked into the graph, per the exporter's metadata.

        Returns None when the export does not say, in which case the decoder falls
        back to inspecting the output.
        """
        raw = self.session.get_modelmeta().custom_metadata_map.get("end2end")
        if raw is None:
            return None
        return str(raw).strip().lower() in ("true", "1", "yes")

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

    def _looks_end2end(self, predictions: np.ndarray) -> bool:
        """Fallback layout sniff for exports whose metadata is silent.

        An end-to-end row ends in an integral class id inside the class range and
        carries a 0..1 score in column 4. A raw row's trailing columns are
        per-class scores, which are almost never all integral.
        """
        if predictions.shape[1] != 6:
            return False
        classes = predictions[:, 5]
        scores = predictions[:, 4]
        integral = np.all(np.isclose(classes, np.round(classes)))
        in_range = bool(classes.min() >= 0 and classes.max() < max(len(self._names), 1))
        scored = bool(scores.min() >= 0.0 and scores.max() <= 1.0)
        return bool(integral and in_range and scored)

    def _decode(
        self, raw: np.ndarray, scale: float, pad_x: int, pad_y: int, height: int, width: int
    ) -> list[Detection]:
        predictions = np.asarray(raw, dtype=np.float32)
        num_classes = len(self._names)
        if predictions.ndim != 2:
            log.warning("unexpected ONNX output shape %s", raw.shape)
            return []

        end2end = self._end2end
        if end2end is None:
            end2end = self._looks_end2end(predictions)
        if end2end:
            return self._decode_end2end(predictions, scale, pad_x, pad_y, height, width)

        # Normalise to (n, 4 + nc).
        if predictions.shape[0] == 4 + num_classes:
            predictions = predictions.transpose(1, 0)
        if predictions.shape[1] < 4 + num_classes:
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


    def _decode_end2end(
        self,
        predictions: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        height: int,
        width: int,
    ) -> list[Detection]:
        """Decode ``[x1, y1, x2, y2, score, class_id]`` rows.

        Already suppressed and sorted by score inside the graph, and padded to a
        fixed row count with near-zero scores, so filtering on the confidence floor
        is all that is needed.
        """
        scores = predictions[:, 4]
        keep = scores >= self.confidence_floor
        if not keep.any():
            return []
        rows = predictions[keep]

        out: list[Detection] = []
        for row in rows:
            label = self._label_map.get(int(round(float(row[5]))))
            if label is None:
                continue
            # Undo the letterbox: subtract padding, divide by the scale, normalise.
            x1 = (float(row[0]) - pad_x) / scale
            y1 = (float(row[1]) - pad_y) / scale
            x2 = (float(row[2]) - pad_x) / scale
            y2 = (float(row[3]) - pad_y) / scale
            if x2 <= x1 or y2 <= y1:
                continue
            box = BBox(x1 / width, y1 / height, x2 / width, y2 / height).clipped()
            if box.area <= 0:
                continue
            out.append(Detection(label=label, confidence=float(row[4]), box=box))
        return out
