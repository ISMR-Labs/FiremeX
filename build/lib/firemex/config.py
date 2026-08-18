"""Configuration: environment-based settings plus the site/camera YAML file.

Two distinct sources, deliberately kept apart:

* :class:`Settings` -- secrets and deployment wiring, from the environment.
* :class:`SiteConfig` -- cameras, contacts and detection tuning, from ``config.yaml``.
  This file is meant to be edited by whoever runs the server and is safe to commit
  to their own private config repo, so no credentials are ever read from it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DetectorBackend = Literal["stub", "ultralytics", "onnx"]
Channel = Literal["call", "sms"]


class Settings(BaseSettings):
    """Deployment settings and secrets, sourced from the environment / .env."""

    model_config = SettingsConfigDict(
        env_prefix="FIREMEX_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # Twilio credentials use their conventional names, not the FIREMEX_ prefix.
    twilio_account_sid: str = Field(default="", validation_alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", validation_alias="TWILIO_AUTH_TOKEN")
    twilio_from_number: str = Field(default="", validation_alias="TWILIO_FROM_NUMBER")

    public_base_url: str = "http://localhost:8000"
    validate_twilio_signature: bool = True

    database_url: str = "sqlite+pysqlite:///./data/firemex.db"
    redis_url: str | None = None

    detector_backend: DetectorBackend = "stub"
    model_path: str = "weights/firemex-yolov26s.pt"
    device: str = "cpu"
    batch_size: int = Field(default=8, ge=1, le=64)
    batch_timeout_ms: int = Field(default=80, ge=1, le=1000)
    image_size: int = Field(default=640, ge=128, le=1920)

    config_path: str = "config.yaml"
    storage_dir: Path = Path("./data")
    log_level: str = "INFO"
    log_json: bool = True

    #: When true the pipeline records incidents but never places a call. This is the
    #: default because an untuned detector on a new site will produce false positives,
    #: and the first thing a false call costs you is the operator's trust.
    shadow_mode: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def clips_dir(self) -> Path:
        return self.storage_dir / "clips"

    @property
    def snapshots_dir(self) -> Path:
        return self.storage_dir / "snapshots"

    def ensure_dirs(self) -> None:
        for path in (self.storage_dir, self.clips_dir, self.snapshots_dir):
            path.mkdir(parents=True, exist_ok=True)

    def twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_number)


class ClassThresholds(BaseModel):
    """Per-class confidence floor. Separate values for fire and smoke because they
    are not equally reliable: smoke is detected earlier but confused more often."""

    fire: float = Field(default=0.40, ge=0.0, le=1.0)
    smoke: float = Field(default=0.45, ge=0.0, le=1.0)

    def for_label(self, label: str) -> float:
        return self.smoke if label == "smoke" else self.fire


class Thresholds(BaseModel):
    day: ClassThresholds = ClassThresholds()
    night: ClassThresholds = ClassThresholds(fire=0.50, smoke=0.55)

    def select(self, is_night: bool) -> ClassThresholds:
        return self.night if is_night else self.day


class ConfirmConfig(BaseModel):
    """Temporal confirmation rules -- how many sampled frames must agree before a
    detection becomes an incident."""

    #: Detections needed within ``window`` sampled frames.
    frames_required: int = Field(default=6, ge=1)
    #: Length of the sliding window, in sampled frames.
    window: int = Field(default=10, ge=1)
    #: Minimum IoU for two boxes to count as "the same thing staying put".
    stability_iou: float = Field(default=0.20, ge=0.0, le=1.0)
    #: How many of the hit boxes must form one spatially stable cluster.
    #: Defaults to ``frames_required`` when omitted.
    stability_frames: int | None = Field(default=None, ge=1)
    #: Require the detected area not to be shrinking across the confirm window.
    #: Cheap flicker filter; the strong growth signal is the longer-window trend
    #: used for severity escalation (see ``growth_window_seconds``).
    require_growth: bool = True
    #: Fractional shrink tolerated by ``require_growth``.
    growth_tolerance: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Long-window area trend used to escalate severity, in seconds.
    growth_window_seconds: float = Field(default=25.0, gt=0)
    #: Drop boxes smaller than this fraction of the frame -- almost always noise.
    min_box_area: float = Field(default=0.0008, ge=0.0, le=1.0)
    #: Close an open incident after this long with no qualifying detection.
    clear_after_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> ConfirmConfig:
        if self.frames_required > self.window:
            raise ValueError("confirm.frames_required cannot exceed confirm.window")
        if self.stability_frames is None:
            self.stability_frames = self.frames_required
        elif self.stability_frames > self.frames_required:
            raise ValueError("confirm.stability_frames cannot exceed confirm.frames_required")
        return self


Point = tuple[float, float]


class CameraConfig(BaseModel):
    id: str
    name: str
    rtsp: str
    location: str = ""
    enabled: bool = True
    #: Inference frames per second. 2-5 is plenty; fire evolves over seconds.
    sample_fps: float = Field(default=3.0, gt=0, le=30)
    thresholds: Thresholds = Thresholds()
    confirm: ConfirmConfig = ConfirmConfig()
    #: Normalised polygons ([[x,y], ...] with x,y in 0..1) whose interiors are
    #: ignored. Use for skylights, stove tops, welding bays, monitors, smoking areas.
    exclude_zones: list[list[Point]] = Field(default_factory=list)
    #: Contact ids to call, in escalation order. Empty means fall back to the
    #: site-wide default chain.
    contacts: list[str] = Field(default_factory=list)
    #: Local clock window treated as night for threshold selection.
    night_start: dt.time = dt.time(18, 30)
    night_end: dt.time = dt.time(6, 30)
    #: Optional lower-resolution substream used for detection, so the expensive
    #: main stream is only decoded when recording an incident clip.
    substream_rtsp: str | None = None

    @field_validator("exclude_zones")
    @classmethod
    def _validate_zones(cls, zones: list[list[Point]]) -> list[list[Point]]:
        for i, zone in enumerate(zones):
            if len(zone) < 3:
                raise ValueError(f"exclude_zones[{i}] needs at least 3 points")
        return zones

    @property
    def detect_url(self) -> str:
        return self.substream_rtsp or self.rtsp

    def is_night(self, at: dt.time) -> bool:
        if self.night_start <= self.night_end:
            return self.night_start <= at < self.night_end
        # Window wraps past midnight.
        return at >= self.night_start or at < self.night_end


class ContactConfig(BaseModel):
    id: str
    name: str
    phone: str
    channels: list[Channel] = Field(default_factory=lambda: ["call", "sms"])
    #: Call attempts for this contact before moving down the chain.
    retries: int = Field(default=2, ge=0, le=10)
    #: Seconds to wait for an acknowledgement before escalating.
    escalate_after_seconds: float = Field(default=45.0, gt=0)

    @field_validator("phone")
    @classmethod
    def _e164(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("+") or not value[1:].replace(" ", "").isdigit():
            raise ValueError(f"phone must be E.164, e.g. +94711234567 (got {value!r})")
        return value.replace(" ", "")


class AlertingConfig(BaseModel):
    #: Grace period before calls go out, during which an operator can cancel from
    #: the dashboard. Set to 0 to alert immediately.
    confirm_delay_seconds: float = Field(default=20.0, ge=0)
    #: Suppress new alerts for the same camera for this long after one fires, so a
    #: ten-minute fire produces one call sequence rather than two hundred.
    cooldown_minutes: float = Field(default=10.0, gt=0)
    #: Fallback escalation chain for cameras that do not name their own contacts.
    default_contacts: list[str] = Field(default_factory=list)
    webhooks: list[str] = Field(default_factory=list)
    #: Spoken alert. ``{camera}``, ``{location}``, ``{site}`` and ``{labels}`` are
    #: substituted. A pre-recorded clip is clearer under stress -- see voice_clip_url.
    voice_template: str = (
        "Fire alert at {site}. {labels} detected on camera {camera}, {location}. "
        "Press 1 to acknowledge this alert."
    )
    #: If set, play this audio URL instead of speaking voice_template.
    voice_clip_url: str | None = None
    sms_template: str = "FiremeX: {labels} detected on {camera} ({location}) at {site}. {link}"


class SiteConfig(BaseModel):
    name: str = "Unnamed site"
    timezone: str = "UTC"
    cameras: list[CameraConfig] = Field(default_factory=list)
    contacts: list[ContactConfig] = Field(default_factory=list)
    alerting: AlertingConfig = AlertingConfig()

    @model_validator(mode="after")
    def _check_references(self) -> SiteConfig:
        camera_ids = [c.id for c in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("duplicate camera id in config")
        contact_ids = {c.id for c in self.contacts}
        if len(contact_ids) != len(self.contacts):
            raise ValueError("duplicate contact id in config")
        for camera in self.cameras:
            for ref in camera.contacts:
                if ref not in contact_ids:
                    raise ValueError(f"camera {camera.id!r} references unknown contact {ref!r}")
        for ref in self.alerting.default_contacts:
            if ref not in contact_ids:
                raise ValueError(f"alerting.default_contacts references unknown contact {ref!r}")
        return self

    def camera(self, camera_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def contact(self, contact_id: str) -> ContactConfig | None:
        return next((c for c in self.contacts if c.id == contact_id), None)

    def escalation_chain(self, camera_id: str) -> list[ContactConfig]:
        """Ordered contacts to call for a camera, falling back to the site default."""
        camera = self.camera(camera_id)
        refs = list(camera.contacts) if camera and camera.contacts else []
        if not refs:
            refs = list(self.alerting.default_contacts)
        return [c for c in (self.contact(r) for r in refs) if c is not None]


def load_site_config(path: str | Path) -> SiteConfig:
    """Read and validate the site YAML. Missing file yields an empty config so the
    server can still boot and be configured through the dashboard."""
    path = Path(path)
    if not path.exists():
        return SiteConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    site = raw.get("site", {}) or {}
    return SiteConfig(
        name=site.get("name", "Unnamed site"),
        timezone=site.get("timezone", "UTC"),
        cameras=raw.get("cameras", []) or [],
        contacts=raw.get("contacts", []) or [],
        alerting=raw.get("alerting", {}) or {},
    )


def dump_site_config(config: SiteConfig, path: str | Path) -> None:
    """Write the site config back to YAML, preserving the on-disk layout."""
    payload = {
        "site": {"name": config.name, "timezone": config.timezone},
        "cameras": [c.model_dump(mode="json", exclude_defaults=True) for c in config.cameras],
        "contacts": [c.model_dump(mode="json", exclude_defaults=True) for c in config.contacts],
        "alerting": config.alerting.model_dump(mode="json", exclude_defaults=True),
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
