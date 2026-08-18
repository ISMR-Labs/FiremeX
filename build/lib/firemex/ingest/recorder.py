"""Incident evidence: an annotated snapshot and a pre/post-event video clip.

Evidence is what turns an alert into something a human can act on in five
seconds. The snapshot is written immediately and synchronously so the SMS link is
live by the time the phone rings; the clip is finished later, once post-roll has
actually been recorded.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..detect.base import Detection

log = logging.getLogger(__name__)

SEVERITY_COLOURS = {"fire": (255, 82, 45), "smoke": (120, 190, 255)}


def annotate(
    image: np.ndarray,
    detections: Sequence[Detection],
    caption: str = "",
) -> np.ndarray:
    """Draw detection boxes and a caption. Returns a new array."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(np.ascontiguousarray(image[:, :, :3]))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, int(min(width, height) * 0.006))

    for detection in detections:
        x1, y1, x2, y2 = detection.box.to_pixels(width, height)
        colour = SEVERITY_COLOURS.get(detection.label, (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)
        label = f"{detection.label} {detection.confidence:.0%}"
        text_y = max(0, y1 - 14)
        # Solid plate behind the text: labels over flame are otherwise unreadable.
        draw.rectangle((x1, text_y, x1 + 8 * len(label), text_y + 14), fill=(0, 0, 0))
        draw.text((x1 + 2, text_y + 2), label, fill=colour)

    if caption:
        draw.rectangle((0, 0, width, 20), fill=(0, 0, 0))
        draw.text((6, 5), caption[:120], fill=(255, 255, 255))
    return np.asarray(canvas)


def write_snapshot(
    path: Path,
    image: np.ndarray,
    detections: Sequence[Detection] = (),
    caption: str = "",
    quality: int = 85,
) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = annotate(image, detections, caption) if (detections or caption) else image[:, :, :3]
    Image.fromarray(np.ascontiguousarray(frame)).save(path, format="JPEG", quality=quality)
    return path


def write_clip(
    path: Path,
    frames: Sequence[tuple[float, np.ndarray]],
    fps: float = 10.0,
    crf: int = 28,
) -> Path | None:
    """Encode frames to H.264 MP4. Returns None when PyAV is unavailable."""
    if not frames:
        return None
    try:
        import av
    except ImportError:
        log.warning("clip not written: install the 'video' extra for PyAV")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0][1].shape[:2]
    # H.264 needs even dimensions.
    width -= width % 2
    height -= height % 2
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=int(round(fps)) or 1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for _, image in frames:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image[:height, :width, :3]), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception:  # noqa: BLE001 - a failed clip must never mask the alert
        log.exception("failed to encode clip %s", path)
        return None
    finally:
        container.close()
    return path


class EvidenceRecorder:
    """Writes snapshots and clips off the event loop."""

    def __init__(self, snapshots_dir: Path, clips_dir: Path, post_roll_seconds: float = 12.0):
        self.snapshots_dir = Path(snapshots_dir)
        self.clips_dir = Path(clips_dir)
        self.post_roll_seconds = post_roll_seconds

    async def capture_snapshot(
        self,
        incident_id: str,
        image: np.ndarray,
        detections: Sequence[Detection],
        caption: str,
    ) -> str | None:
        path = self.snapshots_dir / f"{incident_id}.jpg"
        try:
            await asyncio.to_thread(write_snapshot, path, image, detections, caption)
        except Exception:  # noqa: BLE001
            log.exception("snapshot for %s failed", incident_id)
            return None
        return str(path)

    async def capture_clip(
        self,
        incident_id: str,
        buffer,
        fps: float,
        pre_roll_from: float,
    ) -> str | None:
        """Wait out the post-roll, then encode pre-roll plus post-roll together."""
        await asyncio.sleep(self.post_roll_seconds)
        frames = buffer.since(pre_roll_from)
        if not frames:
            return None
        path = self.clips_dir / f"{incident_id}.mp4"
        try:
            written = await asyncio.to_thread(write_clip, path, frames, fps)
        except Exception:  # noqa: BLE001
            log.exception("clip for %s failed", incident_id)
            return None
        return str(written) if written else None
