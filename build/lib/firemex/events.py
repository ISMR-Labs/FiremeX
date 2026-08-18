"""In-process pub/sub for pushing live updates to dashboard WebSockets."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, queue_size: int = 64) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._queue_size = queue_size

    def publish(self, event: dict) -> None:
        """Fan an event out to subscribers, never blocking on a slow one.

        A dashboard tab that has stopped reading gets its oldest event dropped
        rather than applying backpressure to the detection pipeline.
        """
        for queue in list(self._subscribers):
            if queue.full():
                # Race with a reader draining the same queue is benign.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
