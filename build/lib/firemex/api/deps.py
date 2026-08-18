"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request, status

from ..config import Settings
from ..supervisor import Supervisor

log = logging.getLogger(__name__)


def get_supervisor(request: Request) -> Supervisor:
    supervisor: Supervisor | None = getattr(request.app.state, "supervisor", None)
    if supervisor is None:  # pragma: no cover - only before startup completes
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="supervisor not ready"
        )
    return supervisor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


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
