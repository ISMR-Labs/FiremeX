"""Shared FastAPI dependencies: authentication, authorisation, CSRF.

Access model:

* Everything under ``/api`` requires a session, except ``/api/health`` (liveness
  probes) and the login endpoints.
* ``/twiml/*`` is never session-authenticated -- Twilio cannot log in. Those
  endpoints are protected by request-signature verification instead.
* ``/metrics`` accepts a session *or* a bearer token, so Prometheus can scrape
  without an interactive login while the endpoint is not left wide open. Camera
  names and incident counts are not information to hand to the internet.
* Unsafe methods additionally require a CSRF token, because the session lives in
  a cookie and this UI can silence a fire alert.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Request, Response, status

from .. import auth
from ..config import Settings
from ..supervisor import Supervisor

log = logging.getLogger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_supervisor(request: Request) -> Supervisor:
    supervisor: Supervisor | None = getattr(request.app.state, "supervisor", None)
    if supervisor is None:  # pragma: no cover - only before startup completes
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="supervisor not ready"
        )
    return supervisor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _client_ip(request: Request) -> str | None:
    # X-Forwarded-For is trusted only because this is meant to sit behind a
    # reverse proxy the operator controls. It is used for the session audit trail,
    # never for an access decision.
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host[:64] if request.client else None


async def resolve_principal(request: Request) -> tuple[auth.Principal, str] | None:
    """Resolve the session cookie, or None when unauthenticated."""
    token = request.cookies.get(auth.SESSION_COOKIE)
    if not token:
        return None
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:  # pragma: no cover
        return None
    return await supervisor.store.resolve_session(auth.hash_session_token(token))


async def current_user(request: Request) -> auth.Principal:
    """Require a valid session, a matching CSRF token on unsafe methods, and a
    password that is not the seeded default."""
    resolved = await resolve_principal(request)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )
    principal, csrf_token = resolved

    if request.method not in SAFE_METHODS:
        header = request.headers.get(auth.CSRF_HEADER)
        if not auth.csrf_matches(csrf_token, header):
            log.warning(
                "CSRF check failed for %s %s (user=%s)",
                request.method,
                request.url.path,
                principal.username,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="invalid or missing CSRF token"
            )

    # The seeded admin/admin account may log in and change its password, nothing
    # else. A fire dashboard left on default credentials is a liability, so this is
    # enforced server-side rather than merely nagged about in the UI.
    if principal.must_change_password and not _is_password_change(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="password_change_required",
        )
    return principal


def _is_password_change(request: Request) -> bool:
    return request.url.path in ("/api/auth/password", "/api/auth/me", "/api/auth/logout")


def require_role(required: str):
    """Dependency factory enforcing a minimum role."""

    async def _dependency(principal: auth.Principal = Depends(current_user)) -> auth.Principal:
        if not principal.can(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"this action requires the {required} role or higher",
            )
        return principal

    return _dependency


require_viewer = require_role(auth.ROLE_VIEWER)
require_operator = require_role(auth.ROLE_OPERATOR)
require_admin = require_role(auth.ROLE_ADMIN)


async def metrics_access(request: Request) -> None:
    """Allow a scrape token or a logged-in session."""
    settings: Settings = request.app.state.settings
    expected = settings.metrics_token
    if expected:
        header = request.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(presented.strip(), expected):
            return
    if await resolve_principal(request) is not None:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="metrics require a session or FIREMEX_METRICS_TOKEN bearer token",
    )


def set_session_cookies(
    response: Response, token: str, csrf_token: str, settings: Settings, ttl_hours: int
) -> None:
    secure = settings.public_base_url.lower().startswith("https")
    max_age = ttl_hours * 3600
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=max_age,
        httponly=True,
        # Lax rather than Strict so that following a link into the dashboard from
        # an SMS alert still arrives logged in. Unsafe methods are covered by the
        # CSRF token, not by the cookie policy.
        samesite="lax",
        secure=secure,
        path="/",
    )
    # Readable by JavaScript on purpose: the browser must echo it back in the
    # X-FiremeX-CSRF header. That is the double-submit pattern.
    response.set_cookie(
        auth.CSRF_COOKIE,
        csrf_token,
        max_age=max_age,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    response.delete_cookie(auth.CSRF_COOKIE, path="/")


async def verify_twilio_signature(request: Request) -> None:
    """Reject inbound webhooks that Twilio did not sign.

    These endpoints acknowledge fire alerts and are necessarily public, so an
    unsigned request must never be able to silence an escalation.
    """
    settings: Settings = request.app.state.settings
    if not settings.validate_twilio_signature:
        log.warning("Twilio signature validation is disabled -- do not run this in production")
        return
    if not settings.twilio_auth_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="twilio auth token not configured",
        )

    from twilio.request_validator import RequestValidator

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="missing signature")

    form = await request.form()
    params = {key: str(value) for key, value in form.items()}
    # Twilio signs the URL it dialled. Behind a proxy that is the configured public
    # base URL, not the internal one the app sees.
    public_base = settings.public_base_url.rstrip("/")
    url = f"{public_base}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    validator = RequestValidator(settings.twilio_auth_token)
    if not validator.validate(url, params, signature):
        log.error("rejected Twilio webhook with invalid signature for %s", url)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid signature")


SupervisorDep = Depends(get_supervisor)
SettingsDep = Depends(get_settings)
