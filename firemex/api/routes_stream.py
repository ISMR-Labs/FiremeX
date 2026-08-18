"""Live camera views for the dashboard.

MJPEG rather than WebRTC or HLS. The frames are already decoded and sitting in
each camera's ring buffer, so serving them as ``multipart/x-mixed-replace`` costs
one JPEG encode and renders in a plain ``<img>`` tag -- no player library, no
transcoding, no MediaMTX required, and it works on a locked-down control-room
machine with no internet.

It is not efficient at scale: every viewer costs an encode per frame. Hence the
frame-rate cap, the viewer cap, and the resolution cap. For a wall of many cameras
at full frame rate, put MediaMTX in front and use WebRTC; this exists so the
dashboard works out of the box.

The overlay draws what the detector currently sees, which is the single most
useful thing when tuning a site: you can watch a threshold or an exclusion zone
take effect live.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from .. import auth
from ..detect.base import BBox, Detection
from ..ingest.recorder import annotate
from ..ingest.sources import _downscale
from ..supervisor import Supervisor
from .deps import get_supervisor, require_viewer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cameras", tags=["live view"])

BOUNDARY = "firemexframe"
#: Concurrent MJPEG viewers across all cameras. Each costs a JPEG encode per
#: frame, so this is a guard against the dashboard starving inference.
MAX_VIEWERS = 12
_viewers = 0


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image[:, :, :3])).save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()


def _latest_detections(supervisor: Supervisor, camera_id: str) -> list[Detection]:
    """The detections from the most recent sampled frame, for the overlay."""
    engine = supervisor.engine.get(camera_id)
    if engine is None:
        return []
    window = getattr(engine, "_window", None)
    if not window:
        return []
    return list(window[-1].detections)


def _prepare(
    supervisor: Supervisor,
    camera_id: str,
    camera_name: str,
    width: int,
    overlay: bool,
    quality: int,
) -> bytes | None:
    image = None
    worker = supervisor.workers.get(camera_id)
    if worker is not None:
        image = worker.buffer.latest()
    if image is None:
        return None
    if width and image.shape[1] > width:
        image = _downscale(image, width)
    if overlay:
        detections = _latest_detections(supervisor, camera_id)
        caption = f"{camera_name}  {time.strftime('%H:%M:%S')}"
        if detections:
            caption += "  |  " + ", ".join(
                f"{d.label} {d.confidence:.0%}" for d in detections[:3]
            )
        image = annotate(image, detections, caption)
    return _encode_jpeg(image, quality)


def _require_camera(supervisor: Supervisor, camera_id: str):
    camera = supervisor.site.camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    return camera


@router.get("/{camera_id}/snapshot.jpg")
async def camera_snapshot(
    camera_id: str,
    width: int = Query(default=640, ge=160, le=1920),
    overlay: bool = True,
    quality: int = Query(default=75, ge=30, le=95),
    supervisor: Supervisor = Depends(get_supervisor),
    _: auth.Principal = Depends(require_viewer),
) -> Response:
    """A single current frame. Used for tiles and as the MJPEG fallback."""
    camera = _require_camera(supervisor, camera_id)
    payload = await asyncio.to_thread(
        _prepare, supervisor, camera_id, camera.name, width, overlay, quality
    )
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no frame available yet -- the camera may be offline",
        )
    return Response(
        content=payload,
        media_type="image/jpeg",
        # A live frame must never be cached, by the browser or an intermediary.
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/{camera_id}/live.mjpg")
async def camera_stream(
    camera_id: str,
    request: Request,
    fps: float = Query(default=4.0, gt=0, le=15),
    width: int = Query(default=640, ge=160, le=1920),
    overlay: bool = True,
    quality: int = Query(default=70, ge=30, le=95),
    supervisor: Supervisor = Depends(get_supervisor),
    _: auth.Principal = Depends(require_viewer),
) -> StreamingResponse:
    """Continuous MJPEG. Renders directly in `<img src="...">`."""
    camera = _require_camera(supervisor, camera_id)
    global _viewers
    if _viewers >= MAX_VIEWERS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"too many live viewers (limit {MAX_VIEWERS}); close a tab and retry",
        )

    async def frames():
        global _viewers
        _viewers += 1
        interval = 1.0 / fps
        blank_streak = 0
        try:
            while True:
                if await request.is_disconnected():
                    break
                started = time.monotonic()
                payload = await asyncio.to_thread(
                    _prepare, supervisor, camera_id, camera.name, width, overlay, quality
                )
                if payload is None:
                    # Camera offline. Send a placeholder rather than closing, so the
                    # tile shows "no signal" and recovers on its own when the
                    # stream comes back.
                    blank_streak += 1
                    payload = await asyncio.to_thread(_offline_frame, camera.name, width, quality)
                    if blank_streak > 600:
                        break
                else:
                    blank_streak = 0
                yield (
                    f"--{BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode() + payload + b"\r\n"
                elapsed = time.monotonic() - started
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a dead viewer must not disturb the pipeline
            log.debug("mjpeg stream for %s ended with an error", camera_id, exc_info=True)
        finally:
            _viewers -= 1

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _offline_frame(camera_name: str, width: int, quality: int) -> bytes:
    """A dark 'no signal' placeholder carrying the camera name."""
    height = max(90, int(width * 9 / 16))
    image = np.full((height, width, 3), 22, dtype=np.uint8)
    image[:, :, 2] = 30
    caption = f"{camera_name}  |  NO SIGNAL  {time.strftime('%H:%M:%S')}"
    return _encode_jpeg(annotate(image, [], caption), quality)


@router.get("/{camera_id}/zones-preview.jpg")
async def zones_preview(
    camera_id: str,
    width: int = Query(default=800, ge=320, le=1920),
    supervisor: Supervisor = Depends(get_supervisor),
    _: auth.Principal = Depends(require_viewer),
) -> Response:
    """A current frame with the exclusion zones drawn on it.

    Editing normalised polygon coordinates blind is how zones end up over the wrong
    part of the frame, so the editor previews them against real footage.
    """
    camera = _require_camera(supervisor, camera_id)

    def render() -> bytes | None:
        worker = supervisor.workers.get(camera_id)
        image = worker.buffer.latest() if worker else None
        if image is None:
            return None
        if image.shape[1] > width:
            image = _downscale(image, width)
        # Reuse the detection annotator by expressing each zone as its bounding
        # box, which is what the operator needs to sanity-check placement.
        overlays = [
            Detection("smoke", 1.0, BBox(*_polygon_bounds(zone)))
            for zone in camera.exclude_zones
        ]
        return _encode_jpeg(
            annotate(image, overlays, f"{camera.name}  |  {len(overlays)} exclusion zone(s)"), 80
        )

    payload = await asyncio.to_thread(render)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no frame available yet"
        )
    return Response(content=payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _polygon_bounds(zone) -> tuple[float, float, float, float]:
    xs = [point[0] for point in zone]
    ys = [point[1] for point in zone]
    return (min(xs), min(ys), max(xs), max(ys))
