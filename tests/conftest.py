from __future__ import annotations

import itertools

import pytest

from firemex.config import (
    AlertingConfig,
    CameraConfig,
    ConfirmConfig,
    ContactConfig,
    Settings,
    SiteConfig,
)
from firemex.detect.base import BBox, Detection, FrameResult


@pytest.fixture
def bbox():
    def _make(x1=0.4, y1=0.4, x2=0.6, y2=0.6):
        return BBox(x1, y1, x2, y2)

    return _make


@pytest.fixture
def camera():
    def _make(**overrides):
        confirm = overrides.pop("confirm", None)
        base = {
            "id": "cam-1",
            "name": "Test Camera",
            "location": "Test bay",
            "rtsp": "rtsp://example.invalid/stream",
            "sample_fps": 3.0,
        }
        base.update(overrides)
        if confirm is not None:
            base["confirm"] = (
                confirm if isinstance(confirm, ConfirmConfig) else ConfirmConfig(**confirm)
            )
        return CameraConfig(**base)

    return _make


@pytest.fixture
def counter_ids():
    """Deterministic incident ids so assertions can name them."""
    counter = itertools.count(1)
    return lambda camera_id: f"{camera_id}-{next(counter)}"


@pytest.fixture
def result_factory():
    """Build FrameResults on a synthetic clock advancing at a fixed rate."""

    def _make(fps: float = 3.0, start: float = 1000.0, camera_id: str = "cam-1"):
        interval = 1.0 / fps
        index = itertools.count()

        def _next(detections=(), is_night=False):
            step = next(index)
            ts = start + step * interval
            return FrameResult(
                camera_id=camera_id,
                monotonic_ts=ts,
                wall_ts=1_700_000_000.0 + ts,
                detections=list(detections),
                is_night=is_night,
            )

        return _next

    return _make


@pytest.fixture
def detection():
    def _make(label="fire", confidence=0.9, box=None):
        return Detection(
            label=label, confidence=confidence, box=box or BBox(0.40, 0.40, 0.60, 0.60)
        )

    return _make


@pytest.fixture
def site():
    def _make(cameras=(), contacts=(), **alerting):
        return SiteConfig(
            name="Test Site",
            timezone="UTC",
            cameras=list(cameras),
            contacts=list(contacts),
            alerting=AlertingConfig(**alerting),
        )

    return _make


@pytest.fixture
def contact():
    def _make(contact_id="primary", **overrides):
        base = {
            "id": contact_id,
            "name": contact_id.title(),
            "phone": "+10000000001",
            "channels": ["call"],
            "retries": 0,
            # Tiny ack window so escalation tests exercise the real wait path
            # without spending real seconds in it.
            "escalate_after_seconds": 0.05,
        }
        base.update(overrides)
        return ContactConfig(**base)

    return _make


@pytest.fixture
def settings(tmp_path):
    """Fully explicit settings.

    Every field that a test asserts on is pinned here, including the Twilio
    credentials, so a developer's real .env or exported environment cannot change
    what the tests mean.
    """
    return Settings(
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_from_number="",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "data",
        config_path=str(tmp_path / "config.yaml"),
        detector_backend="stub",
        shadow_mode=True,
        redis_url=None,
        validate_twilio_signature=False,
        public_base_url="http://testserver",
        log_json=False,
    )
