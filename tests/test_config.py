"""Config validation. Bad config must fail loudly at load, not silently at 3am."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from firemex.config import (
    CameraConfig,
    ConfirmConfig,
    ContactConfig,
    SiteConfig,
    dump_site_config,
    load_site_config,
)

YAML = """
site:
  name: "Colombo Warehouse 3"
  timezone: "Asia/Colombo"

cameras:
  - id: loading-bay
    name: "Loading Bay"
    location: "Ground floor, east"
    rtsp: "rtsp://user:pass@192.168.1.41:554/Streaming/Channels/102"
    sample_fps: 3
    thresholds:
      day: {fire: 0.40, smoke: 0.45}
      night: {fire: 0.50, smoke: 0.55}
    confirm:
      frames_required: 6
      window: 10
      require_growth: true
    exclude_zones:
      - [[0.0, 0.0], [0.3, 0.0], [0.3, 0.2], [0.0, 0.2]]
    contacts: [security-desk]

contacts:
  - id: security-desk
    name: "Security Desk"
    phone: "+94711234567"
    channels: [call, sms]
    retries: 2

alerting:
  confirm_delay_seconds: 20
  cooldown_minutes: 10
  default_contacts: [security-desk]
"""


def test_loads_the_documented_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(YAML)
    site = load_site_config(path)

    assert site.name == "Colombo Warehouse 3"
    assert site.timezone == "Asia/Colombo"
    camera = site.camera("loading-bay")
    assert camera is not None
    assert camera.sample_fps == 3
    assert camera.thresholds.night.fire == pytest.approx(0.50)
    assert camera.confirm.frames_required == 6
    assert len(camera.exclude_zones) == 1
    assert site.escalation_chain("loading-bay")[0].id == "security-desk"


def test_missing_file_yields_an_empty_config_so_the_server_still_boots(tmp_path):
    site = load_site_config(tmp_path / "nope.yaml")
    assert site.cameras == []
    assert site.contacts == []


def test_roundtrip_through_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(YAML)
    original = load_site_config(path)

    out = tmp_path / "written.yaml"
    dump_site_config(original, out)
    reloaded = load_site_config(out)

    assert reloaded.name == original.name
    assert [c.id for c in reloaded.cameras] == [c.id for c in original.cameras]
    assert reloaded.camera("loading-bay").exclude_zones == original.camera(
        "loading-bay"
    ).exclude_zones
    assert reloaded.contact("security-desk").phone == "+94711234567"


# ---- reference integrity -------------------------------------------------


def test_camera_referencing_an_unknown_contact_is_rejected():
    with pytest.raises(ValidationError, match="unknown contact"):
        SiteConfig(
            cameras=[
                CameraConfig(id="c1", name="C1", rtsp="rtsp://x", contacts=["ghost"]),
            ],
            contacts=[],
        )


def test_duplicate_camera_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate camera id"):
        SiteConfig(
            cameras=[
                CameraConfig(id="c1", name="A", rtsp="rtsp://a"),
                CameraConfig(id="c1", name="B", rtsp="rtsp://b"),
            ]
        )


def test_duplicate_contact_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate contact id"):
        SiteConfig(
            contacts=[
                ContactConfig(id="x", name="X", phone="+10000000001"),
                ContactConfig(id="x", name="Y", phone="+10000000002"),
            ]
        )


def test_default_contacts_must_exist():
    with pytest.raises(ValidationError, match="default_contacts"):
        SiteConfig(contacts=[], alerting={"default_contacts": ["ghost"]})


# ---- confirmation tuning -------------------------------------------------


def test_frames_required_cannot_exceed_the_window():
    with pytest.raises(ValidationError, match="cannot exceed"):
        ConfirmConfig(frames_required=12, window=10)


def test_stability_frames_defaults_to_frames_required():
    assert ConfirmConfig(frames_required=5, window=10).stability_frames == 5


def test_stability_frames_cannot_exceed_frames_required():
    with pytest.raises(ValidationError, match="stability_frames"):
        ConfirmConfig(frames_required=4, window=10, stability_frames=6)


# ---- phone numbers -------------------------------------------------------


@pytest.mark.parametrize("phone", ["0711234567", "94711234567", "+9471123456a", "phone"])
def test_non_e164_numbers_are_rejected(phone):
    """A malformed number is a contact who will never be reached."""
    with pytest.raises(ValidationError, match="E.164"):
        ContactConfig(id="x", name="X", phone=phone)


def test_spaces_are_stripped_from_numbers():
    assert ContactConfig(id="x", name="X", phone="+94 71 123 4567").phone == "+94711234567"


# ---- zones ---------------------------------------------------------------


def test_a_zone_needs_at_least_three_points():
    with pytest.raises(ValidationError, match="at least 3 points"):
        CameraConfig(
            id="c1", name="C1", rtsp="rtsp://x", exclude_zones=[[(0.0, 0.0), (1.0, 1.0)]]
        )


# ---- day / night switching ----------------------------------------------


def test_night_window_wrapping_past_midnight():
    camera = CameraConfig(
        id="c1",
        name="C1",
        rtsp="rtsp://x",
        night_start=dt.time(18, 30),
        night_end=dt.time(6, 30),
    )
    assert camera.is_night(dt.time(20, 0))
    assert camera.is_night(dt.time(2, 0))
    assert camera.is_night(dt.time(18, 30))
    assert not camera.is_night(dt.time(12, 0))
    assert not camera.is_night(dt.time(6, 30))


def test_night_window_within_one_day():
    camera = CameraConfig(
        id="c1", name="C1", rtsp="rtsp://x", night_start=dt.time(1, 0), night_end=dt.time(5, 0)
    )
    assert camera.is_night(dt.time(3, 0))
    assert not camera.is_night(dt.time(23, 0))


# ---- substream preference -----------------------------------------------


def test_detection_prefers_the_substream():
    """Detection runs at 640px, so decoding 4K for it wastes the whole CPU budget."""
    camera = CameraConfig(
        id="c1", name="C1", rtsp="rtsp://main", substream_rtsp="rtsp://sub"
    )
    assert camera.detect_url == "rtsp://sub"
    assert CameraConfig(id="c2", name="C2", rtsp="rtsp://main").detect_url == "rtsp://main"


# ---- escalation chain resolution ----------------------------------------


def test_chain_falls_back_to_the_site_default():
    site = SiteConfig(
        cameras=[CameraConfig(id="c1", name="C1", rtsp="rtsp://x")],
        contacts=[ContactConfig(id="fallback", name="F", phone="+10000000001")],
        alerting={"default_contacts": ["fallback"]},
    )
    assert [c.id for c in site.escalation_chain("c1")] == ["fallback"]


def test_chain_for_an_unknown_camera_uses_the_default():
    site = SiteConfig(
        contacts=[ContactConfig(id="fallback", name="F", phone="+10000000001")],
        alerting={"default_contacts": ["fallback"]},
    )
    assert [c.id for c in site.escalation_chain("ghost")] == ["fallback"]


def test_chain_preserves_configured_order():
    contacts = [
        ContactConfig(id="a", name="A", phone="+10000000001"),
        ContactConfig(id="b", name="B", phone="+10000000002"),
        ContactConfig(id="c", name="C", phone="+10000000003"),
    ]
    site = SiteConfig(
        cameras=[CameraConfig(id="c1", name="C1", rtsp="rtsp://x", contacts=["c", "a", "b"])],
        contacts=contacts,
    )
    assert [c.id for c in site.escalation_chain("c1")] == ["c", "a", "b"]
