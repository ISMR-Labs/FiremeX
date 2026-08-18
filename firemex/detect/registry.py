"""Detector backend selection."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings
from .base import Detector
from .stub import StubDetector

log = logging.getLogger(__name__)


def build_detector(settings: Settings) -> Detector:
    backend = settings.detector_backend
    if backend == "stub":
        log.warning(
            "using the stub heuristic detector -- development only, it will "
            "false-positive on sunsets and headlights"
        )
        return StubDetector()

    if not Path(settings.model_path).exists():
        raise FileNotFoundError(
            f"model weights not found at {settings.model_path!r}. "
            "Download them with `firemex download-weights`, or set "
            "FIREMEX_DETECTOR_BACKEND=stub to run without a model."
        )

    if backend == "ultralytics":
        from .ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(
            model_path=settings.model_path,
            device=settings.device,
            image_size=settings.image_size,
        )

    if backend == "onnx":
        from .onnx_backend import OnnxDetector

        return OnnxDetector(
            model_path=settings.model_path,
            device=settings.device,
            image_size=settings.image_size,
        )

    raise ValueError(f"unknown detector backend {backend!r}")
