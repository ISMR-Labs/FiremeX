"""Frame sources.

RTSP is decoded with PyAV (FFmpeg bindings) rather than ``cv2.VideoCapture``.
OpenCV's RTSP handling stalls silently and reconnects badly -- a stream that has
died reads as a stream with no motion, which on a fire detector means a camera
that has stopped protecting anything while still looking healthy. PyAV surfaces
real errors, real timestamps, and hardware decode.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


class StreamError(RuntimeError):
    """The stream failed and needs reopening."""


@runtime_checkable
class FrameSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class RtspSource:
    """Blocking RTSP reader. Owned by exactly one decode thread."""

    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        timeout_seconds: float = 8.0,
        max_width: int = 1280,
    ) -> None:
        self.url = url
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        #: Downscale wide streams on decode. 4K H.265 decode can become the
        #: bottleneck long before inference does, and the detector runs at 640px.
        self.max_width = max_width
        self._container = None
        self._stream = None
        self._frames = None

    def open(self) -> None:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "RTSP ingest needs the 'video' extra: pip install 'firemex[video]'"
            ) from exc

        options = {
            # TCP by default: UDP loses packets under load and yields corrupt frames
            # that look like smoke to a detector.
            "rtsp_transport": self.transport,
            "stimeout": str(int(self.timeout_seconds * 1_000_000)),  # microseconds
            "max_delay": "500000",
            "reorder_queue_size": "0",
            "fflags": "nobuffer",
        }
        try:
            self._container = av.open(self.url, options=options, timeout=self.timeout_seconds)
            self._stream = self._container.streams.video[0]
            # Decode only keyframe-anchored output as fast as possible; we sample
            # a few fps anyway, so frame-perfect ordering is not needed.
            self._stream.thread_type = "AUTO"
            self._frames = self._container.decode(self._stream)
        except Exception as exc:  # noqa: BLE001 - any open failure means retry
            self.close()
            raise StreamError(f"could not open {_redact(self.url)}: {exc}") from exc
        log.info("opened stream %s", _redact(self.url))

    def read(self) -> np.ndarray | None:
        if self._frames is None:
            raise StreamError("read() before open()")
        try:
            frame = next(self._frames)
        except StopIteration:
            raise StreamError("stream ended") from None
        except Exception as exc:  # noqa: BLE001 - decode errors mean reopen
            raise StreamError(f"decode failed: {exc}") from exc
        image = frame.to_ndarray(format="rgb24")
        if self.max_width and image.shape[1] > self.max_width:
            image = _downscale(image, self.max_width)
        return image

    def close(self) -> None:
        container, self._container = self._container, None
        self._stream = None
        self._frames = None
        if container is not None:
            try:
                container.close()
            except Exception:  # noqa: BLE001 - closing a broken container often throws
                log.debug("error closing container for %s", _redact(self.url), exc_info=True)


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    """Nearest-neighbour downscale by index striding. No SciPy, no OpenCV."""
    height, width = image.shape[:2]
    scale = max_width / width
    new_w, new_h = max_width, max(1, int(round(height * scale)))
    rows = (np.arange(new_h) / scale).astype(np.int32).clip(0, height - 1)
    cols = (np.arange(new_w) / scale).astype(np.int32).clip(0, width - 1)
    return image[rows][:, cols]


def _redact(url: str) -> str:
    """Strip credentials from an RTSP URL before it reaches a log line."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}" if scheme else f"***@{host}"


class SyntheticSource:
    """Generates a scene with a growing fire-coloured plume.

    Drives demos, integration tests and CI end to end without a camera or a model.
    The plume grows over ``ramp_seconds`` so the growth and persistence rules are
    genuinely exercised rather than bypassed.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps: float = 10.0,
        ignite_after: float = 3.0,
        ramp_seconds: float = 20.0,
        smoke: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.ignite_after = ignite_after
        self.ramp_seconds = ramp_seconds
        self.smoke = smoke
        self._started = 0.0
        self._frame_index = 0

    def open(self) -> None:
        self._started = time.monotonic()
        self._frame_index = 0

    def close(self) -> None:
        return None

    def read(self) -> np.ndarray | None:
        # Pace to the nominal fps so the sampler behaves as it would on a camera.
        target = self._started + self._frame_index / self.fps
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._frame_index += 1

        elapsed = time.monotonic() - self._started
        image = np.full((self.height, self.width, 3), 42, dtype=np.uint8)
        # A dim static background so the frame is not uniformly flat.
        image[:, :, 2] = 58
        image[self.height // 2 :, :, :] = 34

        if elapsed < self.ignite_after:
            return image

        progress = min(1.0, (elapsed - self.ignite_after) / max(self.ramp_seconds, 0.001))
        radius = int(12 + progress * min(self.width, self.height) * 0.28)
        cx, cy = int(self.width * 0.62), int(self.height * 0.62)
        flicker = 0.9 + 0.1 * math.sin(elapsed * 9.0)

        ys, xs = np.ogrid[: self.height, : self.width]
        distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        fire = distance <= radius * flicker
        image[fire] = (250, 120, 30)
        core = distance <= radius * 0.45 * flicker
        image[core] = (255, 225, 120)

        if self.smoke:
            plume = (distance > radius) & (distance <= radius * 2.1) & (ys < cy)
            image[plume] = (150, 150, 152)
        return image
