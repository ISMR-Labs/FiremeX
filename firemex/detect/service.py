"""Batched inference service.

One detector instance serves every camera. Frames from all cameras are collected
into a single batch, because a batch of eight frames costs barely more than one on
a GPU -- batching across cameras is where multi-camera throughput actually comes
from. Running a model per camera wastes VRAM and scales badly.

The submission queue is deliberately bounded. When inference falls behind, callers
are told immediately so they can drop the frame; the alternative is an
ever-growing queue that turns into ever-growing alert latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import metrics
from .base import Detector, Frame, FrameResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Job:
    frame: Frame
    future: asyncio.Future[FrameResult]


class InferenceService:
    def __init__(
        self,
        detector: Detector,
        batch_size: int = 8,
        batch_timeout_ms: int = 80,
        queue_size: int | None = None,
        workers: int = 1,
    ) -> None:
        self.detector = detector
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        # Two batches of headroom: enough to keep the GPU fed, small enough that a
        # stalled detector is noticed in well under a second.
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=queue_size or batch_size * 2)
        # A single worker thread by default: GPU inference serialises anyway, and
        # one thread keeps ordering and warm-up simple.
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="firemex-infer")
        self._task: asyncio.Task | None = None
        self._closing = False
        self.dropped = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.detector.warmup)
        log.info(
            "inference service started backend=%s batch_size=%d timeout=%dms",
            self.detector.name,
            self.batch_size,
            int(self.batch_timeout * 1000),
        )
        self._task = asyncio.create_task(self._run(), name="firemex-inference")

    async def stop(self) -> None:
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Drain anything still waiting so no caller is left hanging on shutdown.
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.cancel()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.detector.close)
        self._executor.shutdown(wait=True)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def submit(self, frame: Frame) -> FrameResult:
        """Queue a frame and wait for its result.

        Raises :class:`asyncio.QueueFull` when the service is saturated. Callers
        should drop the frame and increment their drop counter rather than wait.
        """
        if self._closing:
            raise RuntimeError("inference service is shutting down")
        future: asyncio.Future[FrameResult] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Job(frame=frame, future=future))
        except asyncio.QueueFull:
            self.dropped += 1
            metrics.FRAMES_DROPPED.labels(camera=frame.camera_id, reason="inference_backlog").inc()
            raise
        metrics.INFERENCE_QUEUE_DEPTH.set(self._queue.qsize())
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = await self._collect_batch()
            if not jobs:
                continue
            images = [job.frame.image for job in jobs]
            started = time.perf_counter()
            try:
                results = await loop.run_in_executor(self._executor, self.detector.predict, images)
            except asyncio.CancelledError:
                for job in jobs:
                    if not job.future.done():
                        job.future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the loop
                log.exception("inference batch failed (%d frames)", len(jobs))
                metrics.INFERENCE_ERRORS.inc()
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metrics.INFERENCE_BATCH_SIZE.observe(len(jobs))
            metrics.INFERENCE_LATENCY.observe(elapsed_ms / 1000.0)
            per_frame_ms = elapsed_ms / max(len(jobs), 1)
            for job, detections in zip(jobs, results, strict=False):
                if job.future.done():
                    continue
                job.future.set_result(
                    FrameResult(
                        camera_id=job.frame.camera_id,
                        monotonic_ts=job.frame.monotonic_ts,
                        wall_ts=job.frame.wall_ts,
                        detections=list(detections),
                        is_night=job.frame.is_night,
                        inference_ms=per_frame_ms,
                        sequence=job.frame.sequence,
                    )
                )

    async def _collect_batch(self) -> list[_Job]:
        """Wait for one job, then briefly grab whatever else is ready.

        The timeout only applies once the batch has started, so a single idle
        camera never waits the full window for company.
        """
        first = await self._queue.get()
        jobs = [first]
        deadline = time.monotonic() + self.batch_timeout
        while len(jobs) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        metrics.INFERENCE_QUEUE_DEPTH.set(self._queue.qsize())
        return [job for job in jobs if not job.future.cancelled()]
