"""Pre-event frame buffer.

An incident clip is only useful if it shows what happened *before* the alarm. The
buffer holds the last N seconds of decoded frames per camera so the recorder can
splice pre-roll onto the post-event footage.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameRingBuffer:
    def __init__(self, seconds: float = 12.0, max_frames: int = 240) -> None:
        self.seconds = seconds
        # Hard frame cap as well as a duration cap: a 25 fps stream over 12 s is
        # 300 uncompressed frames, which is real memory per camera.
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=max_frames)

    def push(self, monotonic_ts: float, image: np.ndarray) -> None:
        self._frames.append((monotonic_ts, image))
        cutoff = monotonic_ts - self.seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def since(self, monotonic_ts: float) -> list[tuple[float, np.ndarray]]:
        return [entry for entry in self._frames if entry[0] >= monotonic_ts]

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        return list(self._frames)

    def latest(self) -> np.ndarray | None:
        return self._frames[-1][1] if self._frames else None

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
