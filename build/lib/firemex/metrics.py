"""Prometheus metrics.

Per-camera fps and per-camera false-positive rate have to be visible, or nobody
will ever tune the thresholds correctly.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

FRAMES_SAMPLED = Counter(
    "firemex_frames_sampled_total", "Frames sampled for inference", ["camera"]
)
FRAMES_DROPPED = Counter(
    "firemex_frames_dropped_total", "Frames dropped before inference", ["camera", "reason"]
)
CAMERA_FPS = Gauge("firemex_camera_fps", "Observed sampled frames per second", ["camera"])
CAMERA_UP = Gauge("firemex_camera_up", "1 when the camera stream is connected", ["camera"])
CAMERA_RECONNECTS = Counter(
    "firemex_camera_reconnects_total", "Stream reconnect attempts", ["camera"]
)
LAST_FRAME_AGE = Gauge(
    "firemex_camera_last_frame_age_seconds", "Seconds since the last decoded frame", ["camera"]
)

INFERENCE_QUEUE_DEPTH = Gauge("firemex_inference_queue_depth", "Frames awaiting inference")
INFERENCE_BATCH_SIZE = Histogram(
    "firemex_inference_batch_size", "Frames per batch", buckets=(1, 2, 4, 6, 8, 12, 16, 24, 32)
)
INFERENCE_LATENCY = Histogram(
    "firemex_inference_seconds",
    "Wall time per inference batch",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)
INFERENCE_ERRORS = Counter("firemex_inference_errors_total", "Failed inference batches")

DETECTIONS = Counter(
    "firemex_detections_total",
    "Detections surviving threshold and zone filters",
    ["camera", "label"],
)
DETECTIONS_SUPPRESSED = Counter(
    "firemex_detections_suppressed_total",
    "Detections rejected before confirmation",
    ["camera", "reason"],
)
INCIDENTS_OPENED = Counter("firemex_incidents_opened_total", "Confirmed incidents", ["camera"])
INCIDENTS_CANCELLED = Counter(
    "firemex_incidents_cancelled_total", "Incidents cancelled by an operator", ["camera"]
)
INCIDENTS_ACTIVE = Gauge("firemex_incidents_active", "Currently open incidents")

ALERTS_SENT = Counter("firemex_alerts_sent_total", "Alert attempts", ["channel", "outcome"])
ALERTS_ACKNOWLEDGED = Counter("firemex_alerts_acknowledged_total", "Alerts acknowledged by a human")
ESCALATIONS = Counter("firemex_escalations_total", "Times the chain moved to the next contact")
LAST_SELF_TEST = Gauge(
    "firemex_last_self_test_timestamp_seconds",
    "Unix time of the most recent alerting self-test. Untested alerting is broken "
    "alerting, so its age is worth alerting on.",
)
ALERT_LATENCY = Histogram(
    "firemex_alert_latency_seconds",
    "Confirmation to first call placed",
    buckets=(1, 5, 10, 20, 30, 60, 120, 300),
)
