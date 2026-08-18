"""Per-camera ingest worker.

One worker per camera, each independent: a camera that dies, hangs, or floods must
not affect any other. Decoding is blocking, so it runs on its own thread and hands
sampled frames to the event loop.

Three properties this worker exists to guarantee:

reconnect
    Cameras drop. Reopen with exponential backoff and jitter, forever, and report
    the state so the dashboard shows a dead camera as dead.
watchdog
    A stream that stops delivering frames without erroring is the dangerous
    failure -- it looks like a quiet room. If no frame arrives within the stall
    timeout, tear the connection down and reopen.
drop-oldest
    The frame queue is bounded and drops the *oldest* frame when full. Under load
    a fire detector must stay current; a backlog just converts overload into alert
    latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable

import numpy as np

from .. import metrics
from ..config import CameraConfig
from ..detect.base import Frame, FrameResult
from .ringbuffer import FrameRingBuffer
from .sources import FrameSource, RtspSource, StreamError

log = logging.getLogger(__name__)

SourceFactory = Callable[[CameraConfig], FrameSource]
ResultHandler = Callable[[FrameResult], Awaitable[None]]
Submitter = Callable[[Frame], Awaitable[FrameResult]]


def default_source_factory(camera: CameraConfig) -> FrameSource:
    return RtspSource(camera.detect_url)


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        submit: Submitter,
        on_result: ResultHandler,
        source_factory: SourceFactory = default_source_factory,
        buffer_seconds: float = 12.0,
        stall_timeout: float = 15.0,
        max_backoff: float = 60.0,
        timezone: dt.tzinfo | None = None,
    ) -> None:
        self.camera = camera
        self._submit = submit
        self._on_result = on_result
        self._source_factory = source_factory
        self.buffer = FrameRingBuffer(seconds=buffer_seconds)
        self.stall_timeout = stall_timeout
        self.max_backoff = max_backoff
        self.timezone = timezone

        self.connected = False
        self.reconnects = 0
        self.frames_decoded = 0
        self.frames_sampled = 0
        self.frames_dropped = 0
        self.last_frame_monotonic: float | None = None
        self.last_error: str | None = None
        self.observed_fps = 0.0

        self._queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tasks: list[asyncio.Task] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        #: Set by the watchdog to force the decode thread to reopen the stream.
        self._reset = threading.Event()
        self._sequence = 0

    # ---- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._decode_loop, name=f"firemex-decode-{self.camera.id}", daemon=True
        )
        self._thread.start()
        self._tasks = [
            asyncio.create_task(self._consume(), name=f"firemex-consume-{self.camera.id}"),
            asyncio.create_task(self._watchdog(), name=f"firemex-watchdog-{self.camera.id}"),
        ]
        log.info(
            "camera %s worker started (%.1f fps sampling)",
            self.camera.id,
            self.camera.sample_fps,
        )

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        thread, self._thread = self._thread, None
        if thread is not None:
            # The decode thread can be blocked in FFmpeg; it is a daemon and checks
            # the stop flag between frames, so a bounded join is enough.
            await asyncio.to_thread(thread.join, 5.0)
        metrics.CAMERA_UP.labels(camera=self.camera.id).set(0)
        self.connected = False
        log.info("camera %s worker stopped", self.camera.id)

    def status(self) -> dict:
        age = (
            time.monotonic() - self.last_frame_monotonic
            if self.last_frame_monotonic is not None
            else None
        )
        return {
            "camera_id": self.camera.id,
            "name": self.camera.name,
            "location": self.camera.location,
            "connected": self.connected,
            "reconnects": self.reconnects,
            "frames_decoded": self.frames_decoded,
            "frames_sampled": self.frames_sampled,
            "frames_dropped": self.frames_dropped,
            "observed_fps": round(self.observed_fps, 2),
            "last_frame_age": round(age, 2) if age is not None else None,
            "last_error": self.last_error,
            "buffered_frames": len(self.buffer),
        }

    # ---- decode thread ----------------------------------------------------

    def _decode_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            source = self._source_factory(self.camera)
            try:
                source.open()
            except (StreamError, RuntimeError) as exc:
                self._record_failure(str(exc))
                # Full jitter: a site rebooting its NVR must not have twenty
                # cameras retry in lockstep.
                delay = random.uniform(0.5, 1.0) * backoff
                log.warning(
                    "camera %s connect failed (%s); retrying in %.1fs", self.camera.id, exc, delay
                )
                if self._stop.wait(delay):
                    break
                backoff = min(backoff * 2, self.max_backoff)
                continue

            self._mark_connected()
            backoff = 1.0
            self._reset.clear()
            try:
                self._read_until_error(source)
            except StreamError as exc:
                self._record_failure(str(exc))
                log.warning("camera %s stream error: %s", self.camera.id, exc)
            except Exception as exc:  # noqa: BLE001 - never kill the ingest thread
                self._record_failure(str(exc))
                log.exception("camera %s decode loop crashed", self.camera.id)
            finally:
                source.close()
                self.connected = False
                metrics.CAMERA_UP.labels(camera=self.camera.id).set(0)
            if not self._stop.is_set():
                self._stop.wait(0.5)

    def _read_until_error(self, source: FrameSource) -> None:
        interval = 1.0 / self.camera.sample_fps
        next_sample = 0.0
        fps_window_start = time.monotonic()
        fps_window_count = 0

        while not self._stop.is_set() and not self._reset.is_set():
            image = source.read()
            if image is None:
                continue
            now = time.monotonic()
            self.frames_decoded += 1
            self.last_frame_monotonic = now
            # Every decoded frame feeds the pre-event buffer even though only a
            # few per second reach the detector: the clip should look like video.
            self.buffer.push(now, image)

            if now < next_sample:
                continue
            next_sample = now + interval
            self.frames_sampled += 1
            fps_window_count += 1
            if now - fps_window_start >= 5.0:
                self.observed_fps = fps_window_count / (now - fps_window_start)
                metrics.CAMERA_FPS.labels(camera=self.camera.id).set(self.observed_fps)
                fps_window_start, fps_window_count = now, 0

            self._sequence += 1
            frame = Frame(
                camera_id=self.camera.id,
                image=image,
                monotonic_ts=now,
                wall_ts=time.time(),
                is_night=self._is_night(),
                sequence=self._sequence,
            )
            metrics.FRAMES_SAMPLED.labels(camera=self.camera.id).inc()
            self._offer(frame)

    def _offer(self, frame: Frame) -> None:
        """Hand a frame to the event loop, dropping the oldest if the queue is full."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._enqueue, frame)

    def _enqueue(self, frame: Frame) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.frames_dropped += 1
                metrics.FRAMES_DROPPED.labels(camera=self.camera.id, reason="queue_full").inc()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(frame)

    def _is_night(self) -> bool:
        now = dt.datetime.now(self.timezone) if self.timezone else dt.datetime.now()
        return self.camera.is_night(now.time())

    def _mark_connected(self) -> None:
        self.connected = True
        self.last_error = None
        self.last_frame_monotonic = time.monotonic()
        metrics.CAMERA_UP.labels(camera=self.camera.id).set(1)

    def _record_failure(self, error: str) -> None:
        self.connected = False
        self.last_error = error
        self.reconnects += 1
        metrics.CAMERA_RECONNECTS.labels(camera=self.camera.id).inc()
        metrics.CAMERA_UP.labels(camera=self.camera.id).set(0)

    # ---- event loop tasks -------------------------------------------------

    async def _consume(self) -> None:
        while True:
            frame = await self._queue.get()
            try:
                result = await self._submit(frame)
            except asyncio.QueueFull:
                self.frames_dropped += 1
                continue
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad frame must not stop the camera
                log.exception("inference failed for camera %s", self.camera.id)
                continue
            try:
                await self._on_result(result)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("result handler failed for camera %s", self.camera.id)

    async def _watchdog(self) -> None:
        """Force a reconnect when a connected stream stops delivering frames."""
        while True:
            await asyncio.sleep(min(self.stall_timeout / 3.0, 5.0))
            last = self.last_frame_monotonic
            if last is None:
                continue
            age = time.monotonic() - last
            metrics.LAST_FRAME_AGE.labels(camera=self.camera.id).set(age)
            if self.connected and age > self.stall_timeout:
                log.error(
                    "camera %s stalled: no frame for %.1fs -- forcing reconnect",
                    self.camera.id,
                    age,
                )
                self.last_error = f"stalled for {age:.0f}s"
                self._reset.set()
                # Reset the clock so we do not fire again while the reopen is in
                # flight; the decode loop clears _reset once reconnected.
                self.last_frame_monotonic = time.monotonic()


def latest_frame(worker: CameraWorker) -> np.ndarray | None:
    return worker.buffer.latest()
