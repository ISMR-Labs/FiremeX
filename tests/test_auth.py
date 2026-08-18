"""Authentication, authorisation and the CSRF boundary.

This UI can silence a fire alert and hand out camera credentials, so the boundary
gets tested as carefully as the detector does.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from firemex import auth
from firemex.api.app import create_app
from firemex.config import CameraConfig, ContactConfig, SiteConfig, dump_site_config
from firemex.detect.stub import ScriptedDetector
from firemex.supervisor import Supervisor

STRONG = "a-long-enough-password"


@pytest.fixture
def app_client(settings, tmp_path):
    site = SiteConfig(
        name="Auth Test",
        cameras=[
            CameraConfig(
                id="cam-1",
                name="Cam One",
                rtsp="rtsp://example.invalid/1",
                username="camuser",
                password="camsecret",
                contacts=["security"],
            )
        ],
        contacts=[ContactConfig(id="security", name="Security", phone="+10000000001")],
        alerting={"default_contacts": ["security"]},
    )
    dump_site_config(site, settings.config_path)
    supervisor = Supervisor(
        settings, site=site, detector=ScriptedDetector(script=[], loop=[])
    )
    app = create_app(settings, supervisor=supervisor, start_supervisor=False)
    with TestClient(app) as client:
        yield client


def pin_csrf(client) -> None:
    client.headers[auth.CSRF_HEADER] = client.cookies.get(auth.CSRF_COOKIE)


def login(client, username="admin", password="admin"):
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    if response.status_code == 200:
        pin_csrf(client)
    return response


def admin(client, password=STRONG):
    """Sign in the seeded admin and clear the forced password change."""
    assert login(client).status_code == 200
    client.post(
        "/api/auth/password", json={"current_password": "admin", "new_password": password}
    )
    pin_csrf(client)
    return client


# ---- password hashing ----------------------------------------------------


def test_hash_is_salted_so_identical_passwords_differ():
    first = auth.hash_password("same password")
    second = auth.hash_password("same password")
    assert first != second, "a missing salt would make the hashes identical"
    assert auth.verify_password("same password", first)
    assert auth.verify_password("same password", second)


def test_hash_does_not_contain_the_password():
    assert "hunter2" not in auth.hash_password("hunter2")


@pytest.mark.parametrize(
    "encoded", ["", "garbage", "scrypt$bad", "md5$1$1$1$aaaa$bbbb", "scrypt$x$y$z$aa$bb"]
)
def test_malformed_hashes_are_rejected_not_crashed(encoded):
    assert auth.verify_password("anything", encoded) is False


def test_empty_password_is_never_valid():
    assert auth.verify_password("", auth.hash_password("real")) is False


@pytest.mark.parametrize("weak", ["", "short", "1234567"])
def test_short_passwords_are_rejected(weak):
    with pytest.raises(auth.AuthError):
        auth.validate_password(weak)


def test_the_default_password_cannot_be_reused():
    """Rejected by the length floor, since "admin" is shorter than the minimum.

    Asserted explicitly because it is the one password guaranteed to be tried, and
    a future change to MIN_PASSWORD_LENGTH must not quietly re-allow it.
    """
    with pytest.raises(auth.AuthError):
        auth.validate_password(auth.DEFAULT_PASSWORD)
    assert len(auth.DEFAULT_PASSWORD) < auth.MIN_PASSWORD_LENGTH


@pytest.mark.parametrize("bad", ["ab", "a" * 33, "has space", "semi;colon", "sql'inject"])
def test_invalid_usernames_are_rejected(bad):
    with pytest.raises(auth.AuthError):
        auth.validate_username(bad)


def test_usernames_are_lowercased():
    assert auth.validate_username("  AdMiN  ") == "admin"


# ---- the login flow ------------------------------------------------------


def test_seeded_admin_can_log_in(app_client):
    response = login(app_client)
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["must_change_password"] is True


def test_session_cookie_is_httponly(app_client):
    response = login(app_client)
    cookies = response.headers.get_list("set-cookie")
    session_cookie = next(c for c in cookies if c.startswith(auth.SESSION_COOKIE))
    assert "HttpOnly" in session_cookie, "the session token must be unreadable by JS"
    # The CSRF cookie is deliberately readable -- the page has to echo it back.
    csrf_cookie = next(c for c in cookies if c.startswith(auth.CSRF_COOKIE))
    assert "HttpOnly" not in csrf_cookie


def test_wrong_password_is_rejected(app_client):
    response = app_client.post(
        "/api/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert response.status_code == 401


def test_unknown_user_gives_the_same_message_as_a_wrong_password(app_client):
    """Distinguishing the two would let anyone enumerate accounts."""
    unknown = app_client.post(
        "/api/auth/login", json={"username": "nobody", "password": "wrong"}
    )
    wrong = app_client.post(
        "/api/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_repeated_failures_lock_the_account(app_client):
    for _ in range(5):
        app_client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    response = app_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 429
    assert "locked" in response.json()["detail"]


# ---- the forced password change -----------------------------------------


def test_default_account_cannot_do_anything_but_change_its_password(app_client):
    """A fire dashboard left on admin/admin is a liability, so this is enforced
    server-side rather than merely nagged about in the UI."""
    login(app_client)
    for path in ("/api/status", "/api/cameras", "/api/users", "/api/incidents"):
        response = app_client.get(path)
        assert response.status_code == 403, path
        assert response.json()["detail"] == "password_change_required"


def test_changing_the_password_unlocks_the_rest_of_the_api(app_client):
    login(app_client)
    response = app_client.post(
        "/api/auth/password", json={"current_password": "admin", "new_password": STRONG}
    )
    assert response.status_code == 200
    pin_csrf(app_client)
    assert app_client.get("/api/status").status_code == 200
    assert app_client.get("/api/auth/me").json()["must_change_password"] is False


def test_password_change_requires_the_current_password(app_client):
    login(app_client)
    response = app_client.post(
        "/api/auth/password", json={"current_password": "nope", "new_password": STRONG}
    )
    assert response.status_code == 403


def test_new_password_must_differ(app_client):
    admin(app_client)
    response = app_client.post(
        "/api/auth/password", json={"current_password": STRONG, "new_password": STRONG}
    )
    assert response.status_code == 400


def test_changing_the_password_keeps_you_signed_in(app_client):
    """Revoke every old session, but issue a fresh one -- an operator should not be
    bounced to the login screen mid-incident."""
    admin(app_client)
    assert app_client.get("/api/auth/me").status_code == 200


# ---- the authentication boundary ----------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/status",
        "/api/stats",
        "/api/cameras",
        "/api/contacts",
        "/api/incidents",
        "/api/users",
        "/api/config",
        "/api/alerting",
        "/api/detection",
        "/api/cameras/cam-1/snapshot.jpg",
        "/api/cameras/cam-1/live.mjpg",
        "/metrics",
    ],
)
def test_endpoints_require_a_session(app_client, path):
    assert app_client.get(path).status_code == 401, path


def test_health_stays_open_for_probes(app_client):
    assert app_client.get("/api/health").status_code == 200


def test_readiness_is_open_but_withholds_detail_from_strangers(app_client):
    """Orchestrators need the verdict; an anonymous caller does not need to be told
    which camera is down."""
    body = app_client.get("/api/ready").json()
    assert "problem_count" in body
    assert "problems" not in body

    admin(app_client)
    body = app_client.get("/api/ready").json()
    assert "problems" in body


def test_logout_invalidates_the_session(app_client):
    admin(app_client)
    assert app_client.post("/api/auth/logout").status_code == 200
    assert app_client.get("/api/status").status_code == 401


def test_a_forged_session_cookie_is_rejected(app_client):
    app_client.cookies.set(auth.SESSION_COOKIE, "not-a-real-token")
    assert app_client.get("/api/status").status_code == 401


# ---- CSRF ----------------------------------------------------------------


def test_unsafe_requests_without_the_csrf_header_are_refused(app_client):
    admin(app_client)
    del app_client.headers[auth.CSRF_HEADER]
    response = app_client.patch("/api/site", json={"name": "Hijacked"})
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


def test_a_wrong_csrf_token_is_refused(app_client):
    admin(app_client)
    app_client.headers[auth.CSRF_HEADER] = "wrong-token"
    assert app_client.patch("/api/site", json={"name": "Hijacked"}).status_code == 403


def test_safe_requests_do_not_need_a_csrf_token(app_client):
    admin(app_client)
    del app_client.headers[auth.CSRF_HEADER]
    assert app_client.get("/api/status").status_code == 200


# ---- roles ---------------------------------------------------------------


def make_user(client, username, role, password=STRONG):
    response = client.post(
        "/api/users", json={"username": username, "password": password, "role": role}
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def as_role(app_client, settings, tmp_path):
    """Sign in as a freshly created user of the given role."""

    def _switch(role: str):
        admin(app_client)
        make_user(app_client, f"{role}user", role)
        app_client.post("/api/auth/logout")
        app_client.cookies.clear()
        assert login(app_client, f"{role}user", STRONG).status_code == 200
        return app_client

    return _switch


def test_viewer_can_read_but_not_change(as_role):
    client = as_role("viewer")
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/incidents").status_code == 200
    # Configuration is admin-only.
    assert client.get("/api/cameras").status_code == 403
    assert client.post("/api/incidents/x/review", json={"verdict": "real"}).status_code == 403


def test_operator_can_act_on_incidents_but_not_reconfigure(as_role):
    client = as_role("operator")
    assert client.get("/api/status").status_code == 200
    # 404, not 403: the role check passed and the incident simply does not exist.
    assert client.post("/api/incidents/ghost/cancel", json={}).status_code == 404
    assert client.get("/api/cameras").status_code == 403
    assert client.get("/api/users").status_code == 403


def test_admin_can_do_everything(app_client):
    admin(app_client)
    assert app_client.get("/api/cameras").status_code == 200
    assert app_client.get("/api/users").status_code == 200
    assert app_client.get("/api/detection").status_code == 200


# ---- user administration -------------------------------------------------


def test_admin_can_create_and_list_users(app_client):
    admin(app_client)
    make_user(app_client, "operator1", "operator")
    usernames = {u["username"] for u in app_client.get("/api/users").json()}
    assert usernames == {"admin", "operator1"}


def test_duplicate_usernames_conflict(app_client):
    admin(app_client)
    make_user(app_client, "dupe", "viewer")
    response = app_client.post(
        "/api/users", json={"username": "dupe", "password": STRONG, "role": "viewer"}
    )
    assert response.status_code == 409


def test_weak_password_is_rejected_on_create(app_client):
    admin(app_client)
    response = app_client.post(
        "/api/users", json={"username": "weak", "password": "short", "role": "viewer"}
    )
    assert response.status_code == 400


def test_unknown_role_is_rejected(app_client):
    admin(app_client)
    response = app_client.post(
        "/api/users", json={"username": "odd", "password": STRONG, "role": "superuser"}
    )
    assert response.status_code == 400


def test_admin_reset_password_forces_a_change_at_next_login(app_client):
    admin(app_client)
    make_user(app_client, "forgetful", "viewer")
    response = app_client.patch(
        "/api/users/forgetful", json={"new_password": "temporary-password"}
    )
    assert response.status_code == 200
    assert response.json()["must_change_password"] is True

    app_client.post("/api/auth/logout")
    app_client.cookies.clear()
    body = login(app_client, "forgetful", "temporary-password").json()
    assert body["must_change_password"] is True


def test_disabled_user_cannot_log_in(app_client):
    admin(app_client)
    make_user(app_client, "gone", "viewer")
    app_client.patch("/api/users/gone", json={"disabled": True})
    app_client.post("/api/auth/logout")
    app_client.cookies.clear()
    assert login(app_client, "gone", STRONG).status_code == 403


def test_the_last_admin_cannot_be_deleted(app_client):
    """An appliance nobody can administer is the wrong thing to discover during an
    incident."""
    admin(app_client)
    make_user(app_client, "other", "viewer")
    # Deleting yourself is refused outright.
    assert app_client.delete("/api/users/admin").status_code == 400

    # And so is demoting the only admin.
    assert app_client.patch("/api/users/admin", json={"role": "viewer"}).status_code == 409
    assert app_client.patch("/api/users/admin", json={"disabled": True}).status_code == 409


def test_second_admin_allows_demoting_the_first(app_client):
    admin(app_client)
    make_user(app_client, "admin2", "admin")
    assert app_client.patch("/api/users/admin", json={"role": "operator"}).status_code == 200


def test_deleting_a_user_ends_their_sessions(app_client):
    admin(app_client)
    make_user(app_client, "temp", "viewer")
    assert app_client.delete("/api/users/temp").status_code == 204
    assert app_client.get("/api/users").status_code == 200


def test_unlock_clears_a_lockout(app_client):
    admin(app_client)
    make_user(app_client, "locked", "viewer")
    other = TestClient(app_client.app)
    for _ in range(5):
        other.post("/api/auth/login", json={"username": "locked", "password": "wrong"})
    assert other.post(
        "/api/auth/login", json={"username": "locked", "password": STRONG}
    ).status_code == 429

    assert app_client.post("/api/users/locked/unlock").status_code == 200
    assert other.post(
        "/api/auth/login", json={"username": "locked", "password": STRONG}
    ).status_code == 200


# ---- credential leakage --------------------------------------------------


def test_camera_password_is_never_returned(app_client):
    """Not even to an authenticated admin: it would end up in browser history,
    screenshots and support tickets."""
    admin(app_client)
    for payload in (
        app_client.get("/api/cameras").json(),
        [app_client.get("/api/config").json()],
    ):
        blob = str(payload)
        assert "camsecret" not in blob
    camera = app_client.get("/api/cameras").json()[0]
    assert "password" not in camera
    assert camera["has_password"] is True
    assert camera["username"] == "camuser"


def test_metrics_token_grants_scrape_access(settings, app_client):
    settings.metrics_token = "scrape-me"
    assert app_client.get("/metrics").status_code == 401
    response = app_client.get("/metrics", headers={"Authorization": "Bearer scrape-me"})
    assert response.status_code == 200
    assert "firemex_frames_sampled_total" in response.text


def test_wrong_metrics_token_is_refused(settings, app_client):
    settings.metrics_token = "scrape-me"
    assert app_client.get(
        "/metrics", headers={"Authorization": "Bearer nope"}
    ).status_code == 401


def test_session_grants_metrics_access_without_a_token(app_client):
    admin(app_client)
    assert app_client.get("/metrics").status_code == 200


# ---- websocket -----------------------------------------------------------


def test_websocket_rejects_an_unauthenticated_client(app_client):
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect) as excinfo,
        app_client.websocket_connect("/api/live") as ws,
    ):
        ws.receive_json()
    assert excinfo.value.code == 1008


def test_websocket_accepts_a_signed_in_client(app_client):
    admin(app_client)
    with app_client.websocket_connect("/api/live") as ws:
        first = ws.receive_json()
    assert first["type"] == "status"


# ---- RTSP URL handling ---------------------------------------------------


def test_credentials_are_percent_encoded_into_the_url():
    """A password containing '@' or ':' must not be able to break the URL."""
    url = auth.build_rtsp_url("rtsp://10.0.0.5:554/stream", "admin", "p@ss:word/x")
    assert url == "rtsp://admin:p%40ss%3Aword%2Fx@10.0.0.5:554/stream"


def test_no_credentials_leaves_the_url_untouched():
    assert auth.build_rtsp_url("rtsp://10.0.0.5/s", None, None) == "rtsp://10.0.0.5/s"


def test_redaction_strips_user_and_password():
    redacted = auth.redact_url("rtsp://admin:secret@10.0.0.5:554/s")
    assert "secret" not in redacted
    assert "admin" not in redacted
    assert "10.0.0.5:554" in redacted


# ---- the session probe ---------------------------------------------------


def test_session_probe_reports_anonymous_without_erroring(app_client):
    """Always 200. The shell calls this on every page load, and an endpoint that
    401s for the normal not-logged-in case trains operators to ignore console
    errors."""
    response = app_client.get("/api/auth/session")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_session_probe_reports_the_signed_in_user(app_client):
    admin(app_client)
    body = app_client.get("/api/auth/session").json()
    assert body["authenticated"] is True
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["is_admin"] is True
    assert body["must_change_password"] is False


def test_session_probe_surfaces_the_forced_password_change(app_client):
    login(app_client)
    body = app_client.get("/api/auth/session").json()
    assert body["authenticated"] is True
    assert body["must_change_password"] is True
