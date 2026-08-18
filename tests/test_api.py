"""HTTP surface, exercised against a real supervisor wired to a stub detector."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from firemex import auth
from firemex.api.app import create_app
from firemex.config import CameraConfig, ContactConfig, SiteConfig, dump_site_config
from firemex.detect.base import BBox, Detection
from firemex.detect.stub import ScriptedDetector
from firemex.incident.engine import Incident, Severity
from firemex.supervisor import Supervisor

ADMIN_PASSWORD = "test-admin-password"


@pytest.fixture
def configured_site(tmp_path, settings):
    site = SiteConfig(
        name="Test Site",
        timezone="UTC",
        cameras=[
            CameraConfig(
                id="cam-1",
                name="Loading Bay",
                location="East",
                rtsp="rtsp://example.invalid/1",
                contacts=["security"],
            )
        ],
        contacts=[ContactConfig(id="security", name="Security", phone="+10000000001")],
        alerting={"default_contacts": ["security"]},
    )
    dump_site_config(site, settings.config_path)
    return site


class IdleSource:
    """A camera that connects and then delivers nothing.

    Lets the supervisor start for real -- inference service, dispatcher, workers --
    without touching the network. The camera reads as connected-but-silent, which
    is also the exact state the stall watchdog exists to catch.
    """

    def open(self) -> None:
        return None

    def read(self):
        time.sleep(0.02)
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def supervisor(settings, configured_site):
    return Supervisor(
        settings,
        site=configured_site,
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: IdleSource(),
    )


@pytest.fixture
def anon_client(settings, supervisor):
    """An unauthenticated client, for testing the auth boundary itself."""
    app = create_app(settings, supervisor=supervisor, start_supervisor=True)
    with TestClient(app) as test_client:
        yield test_client


def sign_in(test_client, username="admin", password="admin", new_password=ADMIN_PASSWORD):
    """Log in and, if the account is on its forced first password, replace it.

    Also pins the CSRF token as a default header. The browser reads the token from
    a JS-readable cookie and echoes it back; TestClient will not do that on its
    own, so tests have to mirror the same double-submit the UI performs.
    """
    response = test_client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    _pin_csrf(test_client)

    if response.json().get("must_change_password"):
        changed = test_client.post(
            "/api/auth/password",
            json={"current_password": password, "new_password": new_password},
        )
        assert changed.status_code == 200, changed.text
        # The password change rotates the session, so re-pin.
        _pin_csrf(test_client)
    return test_client


def _pin_csrf(test_client) -> None:
    token = test_client.cookies.get(auth.CSRF_COOKIE)
    assert token, "login should have set the CSRF cookie"
    test_client.headers[auth.CSRF_HEADER] = token


@pytest.fixture
def client(anon_client):
    """Signed in as the seeded admin, past the forced password change."""
    return sign_in(anon_client)


# ---- status and health ---------------------------------------------------


def test_health_is_trivially_alive(anon_client):
    """Liveness stays open so orchestrators can probe without credentials."""
    response = anon_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_the_site_and_cameras(client):
    payload = client.get("/api/status").json()
    assert payload["site"] == "Test Site"
    assert payload["shadow_mode"] is True
    assert len(payload["cameras"]) == 1
    assert payload["cameras"][0]["camera_id"] == "cam-1"


def test_readiness_is_ok_when_the_camera_is_connected(client):
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "problems": []}


def test_readiness_fails_while_a_camera_is_down(settings, configured_site):
    """A disconnected camera is real degradation, and readiness must say so rather
    than reporting a healthy system that is watching nothing."""
    from firemex.ingest.sources import StreamError

    class DeadSource:
        def open(self):
            raise StreamError("connection refused")

        def read(self):
            raise StreamError("not open")

        def close(self):
            return None

    supervisor = Supervisor(
        settings,
        site=configured_site,
        detector=ScriptedDetector(script=[], loop=[]),
        source_factory=lambda camera: DeadSource(),
    )
    app = create_app(settings, supervisor=supervisor, start_supervisor=True)
    with TestClient(app) as client:
        sign_in(client)
        # Let the decode thread attempt its first connection and fail.
        time.sleep(0.15)
        response = client.get("/api/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert any("cam-1" in problem for problem in body["problems"])
        assert any("connection refused" in problem for problem in body["problems"])


def test_metrics_are_exported_in_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "firemex_frames_sampled_total" in response.text


def test_config_endpoint_never_leaks_secrets(client):
    body = client.get("/api/config").json()
    serialised = str(body)
    assert "auth_token" not in serialised
    assert "twilio_account_sid" not in serialised
    assert body["runtime"]["twilio_configured"] is False


# ---- camera CRUD ---------------------------------------------------------


def test_list_cameras(client):
    cameras = client.get("/api/cameras").json()
    assert [c["id"] for c in cameras] == ["cam-1"]


def test_create_camera_persists_to_the_config_file(client, settings):
    body = {
        "id": "cam-2",
        "name": "Kitchen",
        "rtsp": "rtsp://example.invalid/2",
        "location": "Level 1",
        "enabled": False,
    }
    response = client.post("/api/cameras", json=body)
    assert response.status_code == 201

    from firemex.config import load_site_config

    reloaded = load_site_config(settings.config_path)
    assert reloaded.camera("cam-2") is not None
    assert reloaded.camera("cam-2").name == "Kitchen"


def test_duplicate_camera_id_conflicts(client):
    body = {"id": "cam-1", "name": "Dup", "rtsp": "rtsp://example.invalid/dup"}
    assert client.post("/api/cameras", json=body).status_code == 409


def test_update_camera_requires_matching_id(client):
    body = {"id": "other", "name": "X", "rtsp": "rtsp://example.invalid/x"}
    assert client.put("/api/cameras/cam-1", json=body).status_code == 400


def test_update_camera_changes_thresholds(client):
    body = {
        "id": "cam-1",
        "name": "Loading Bay",
        "rtsp": "rtsp://example.invalid/1",
        "thresholds": {"day": {"fire": 0.7, "smoke": 0.75}},
        "contacts": ["security"],
    }
    response = client.put("/api/cameras/cam-1", json=body)
    assert response.status_code == 200
    assert response.json()["thresholds"]["day"]["fire"] == pytest.approx(0.7)


def test_delete_unknown_camera_is_404(client):
    assert client.delete("/api/cameras/ghost").status_code == 404


def test_camera_referencing_an_unknown_contact_is_rejected(client):
    body = {
        "id": "cam-9",
        "name": "Bad",
        "rtsp": "rtsp://example.invalid/9",
        "contacts": ["nobody"],
    }
    response = client.post("/api/cameras", json=body)
    assert response.status_code == 400
    assert "unknown contact" in response.json()["detail"]


def test_invalid_zone_is_rejected(client):
    body = {
        "id": "cam-8",
        "name": "Bad zone",
        "rtsp": "rtsp://example.invalid/8",
        "exclude_zones": [[[0.0, 0.0], [1.0, 1.0]]],
    }
    assert client.post("/api/cameras", json=body).status_code == 422


# ---- contact CRUD -------------------------------------------------------


def test_create_contact(client):
    body = {"id": "owner", "name": "Owner", "phone": "+10000000009"}
    assert client.post("/api/contacts", json=body).status_code == 201
    assert {c["id"] for c in client.get("/api/contacts").json()} == {"security", "owner"}


def test_create_contact_with_a_bad_number_is_rejected(client):
    body = {"id": "bad", "name": "Bad", "phone": "0711234567"}
    assert client.post("/api/contacts", json=body).status_code == 422


def test_deleting_a_contact_cleans_up_dangling_references(client, settings):
    """Deleting the last contact a camera names must not leave an invalid config."""
    response = client.delete("/api/contacts/security")
    assert response.status_code == 204

    from firemex.config import load_site_config

    reloaded = load_site_config(settings.config_path)
    assert reloaded.contacts == []
    assert reloaded.camera("cam-1").contacts == []
    assert reloaded.alerting.default_contacts == []


def test_test_call_without_twilio_is_503_not_a_silent_noop(client):
    response = client.post("/api/contacts/security/test-call")
    assert response.status_code == 503
    assert "Twilio" in response.json()["detail"]


def test_test_call_for_an_unknown_contact_is_404(client):
    assert client.post("/api/contacts/ghost/test-call").status_code == 404


# ---- site ---------------------------------------------------------------


def test_patch_site_name(client):
    response = client.patch("/api/site", json={"name": "Renamed Site"})
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed Site"
    assert client.get("/api/status").json()["site"] == "Renamed Site"


# ---- incidents ----------------------------------------------------------


def make_incident(incident_id="inc-1"):
    return Incident(
        id=incident_id,
        camera_id="cam-1",
        opened_monotonic=10.0,
        # Recent, so it falls inside the stats window.
        opened_wall=time.time(),
        severity=Severity.CRITICAL,
        labels={"fire", "smoke"},
        peak_confidence=0.93,
        envelope=BBox(0.4, 0.4, 0.6, 0.6),
        frames_confirmed=6,
        last_hit_monotonic=10.0,
    )


@pytest.fixture
def stored_incident(client):
    supervisor = client.app.state.supervisor
    incident = make_incident()
    asyncio.run(
        supervisor.store.save_incident(
            incident,
            camera_name="Loading Bay",
            location="East",
            detections=[Detection("fire", 0.93, BBox(0.4, 0.4, 0.6, 0.6))],
            shadow_mode=True,
        )
    )
    return incident


def test_incident_list_and_detail(client, stored_incident):
    listing = client.get("/api/incidents").json()
    assert [item["id"] for item in listing] == ["inc-1"]

    detail = client.get("/api/incidents/inc-1").json()
    assert detail["camera_name"] == "Loading Bay"
    assert detail["severity"] == "critical"
    assert sorted(detail["labels"].split(",")) == ["fire", "smoke"]
    assert detail["detections"][0]["label"] == "fire"


def test_unknown_incident_is_404(client):
    assert client.get("/api/incidents/ghost").status_code == 404


def test_review_records_the_operator_verdict(client, stored_incident):
    """The false-positive verdict is what feeds fine-tuning, so it must persist."""
    response = client.post(
        "/api/incidents/inc-1/review",
        json={"verdict": "false_positive", "note": "sunset through the skylight"},
    )
    assert response.status_code == 200
    assert response.json()["review"] == "false_positive"
    note = client.get("/api/incidents/inc-1").json()["review_note"]
    assert "sunset through the skylight" in note
    # Attributed: a false-positive label feeds fine-tuning, so who judged it matters
    # when the labels are later disputed.
    assert "[admin]" in note


def test_review_rejects_an_unknown_verdict(client, stored_incident):
    response = client.post("/api/incidents/inc-1/review", json={"verdict": "maybe"})
    assert response.status_code == 422


def test_unreviewed_filter(client, stored_incident):
    assert len(client.get("/api/incidents?unreviewed_only=true").json()) == 1
    client.post("/api/incidents/inc-1/review", json={"verdict": "real"})
    assert client.get("/api/incidents?unreviewed_only=true").json() == []


def test_cancel_marks_the_incident_a_false_positive(client, stored_incident):
    response = client.post("/api/incidents/inc-1/cancel", json={"reason": "no fire, checked"})
    assert response.status_code == 200
    detail = client.get("/api/incidents/inc-1").json()
    assert detail["review"] == "false_positive"


def test_cancelling_an_unknown_incident_is_404(client):
    assert client.post("/api/incidents/ghost/cancel", json={}).status_code == 404


def test_missing_snapshot_is_404_not_a_broken_image(client, stored_incident):
    assert client.get("/api/incidents/inc-1/snapshot").status_code == 404


def test_stats_report_the_review_queue(client, stored_incident):
    stats = client.get("/api/stats?days=7").json()
    assert stats["incidents"] == 1
    assert stats["unreviewed"] == 1
    # Rate is None until something has actually been reviewed; reporting 0% for an
    # unreviewed queue would hide an untuned detector.
    assert stats["false_positive_rate"] is None

    client.post("/api/incidents/inc-1/review", json={"verdict": "false_positive"})
    stats = client.get("/api/stats?days=7").json()
    assert stats["false_positives"] == 1
    assert stats["false_positive_rate"] == pytest.approx(1.0)


# ---- Twilio webhooks ----------------------------------------------------


def test_alert_twiml_speaks_the_incident_details(client, stored_incident):
    response = client.post("/twiml/alert/inc-1?contact=security")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "Loading Bay" in response.text
    assert "<Gather" in response.text


def test_alert_twiml_for_an_unknown_incident_says_something_true(client):
    """A self-test or purged incident must not narrate a fire that isn't there."""
    response = client.post("/twiml/alert/selftest-security")
    assert response.status_code == 200
    assert "test call" in response.text.lower()


def test_pressing_one_acknowledges(client, stored_incident):
    response = client.post("/twiml/ack/inc-1?contact=security", data={"Digits": "1"})
    assert response.status_code == 200
    assert "Acknowledged" in response.text
    assert client.get("/api/incidents/inc-1").json()["acknowledged_by"] == "security"


def test_pressing_anything_else_does_not_acknowledge(client, stored_incident):
    response = client.post("/twiml/ack/inc-1?contact=security", data={"Digits": "7"})
    assert "not a valid response" in response.text
    assert client.get("/api/incidents/inc-1").json()["acknowledged_by"] is None


def test_no_digit_does_not_acknowledge(client, stored_incident):
    client.post("/twiml/ack/inc-1?contact=security", data={})
    assert client.get("/api/incidents/inc-1").json()["acknowledged_by"] is None


def test_call_status_callback_is_accepted_but_does_not_acknowledge(client, stored_incident):
    """Call progress says the phone was answered, not that a human understood."""
    response = client.post(
        "/twiml/status/inc-1?contact=security",
        data={"CallStatus": "completed", "CallSid": "CA123"},
    )
    assert response.status_code == 204
    assert client.get("/api/incidents/inc-1").json()["acknowledged_by"] is None


# ---- signature enforcement ----------------------------------------------


def test_unsigned_webhooks_are_rejected_when_validation_is_on(settings, supervisor):
    """These endpoints can silence a fire alert, so an unsigned request must fail."""
    settings.validate_twilio_signature = True
    settings.twilio_auth_token = "test-token"
    app = create_app(settings, supervisor=supervisor, start_supervisor=True)
    with TestClient(app) as client:
        assert client.post("/twiml/ack/inc-1", data={"Digits": "1"}).status_code == 403
        assert client.post("/twiml/alert/inc-1").status_code == 403


# ---- dashboard ----------------------------------------------------------


def test_dashboard_is_served(anon_client):
    response = anon_client.get("/")
    assert response.status_code == 200
    assert "FiremeX" in response.text


def test_static_assets_are_served(anon_client):
    """Served unauthenticated: the shell decides whether to show the login form,
    and every byte of real data behind it needs a session."""
    assert anon_client.get("/static/js/main.js").status_code == 200
    assert anon_client.get("/static/js/views/cameras.js").status_code == 200
    assert anon_client.get("/static/style.css").status_code == 200


def test_openapi_schema_builds(anon_client):
    schema = anon_client.get("/openapi.json").json()
    assert "/api/incidents" in schema["paths"]
    assert "/twiml/ack/{incident_id}" in schema["paths"]
    assert "/api/auth/login" in schema["paths"]


