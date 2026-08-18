"""Status, health, metrics and configuration reload."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import Settings
from ..supervisor import Supervisor
from .deps import (
    get_settings,
    get_supervisor,
    metrics_access,
    require_admin,
    require_viewer,
    resolve_principal,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/api/status", dependencies=[Depends(require_viewer)])
async def get_status(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    return supervisor.status()


@router.get("/api/stats", dependencies=[Depends(require_viewer)])
async def get_stats(days: int = 7, supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    return await supervisor.store.stats(days=days)


@router.get("/api/health")
async def health() -> dict:
    """Liveness only: is the process serving? Kept trivial on purpose."""
    return {"status": "ok"}


@router.get("/api/ready")
async def ready(
    request: Request, response: Response, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    """Readiness. A disconnected camera is a real degradation and reports as one.

    Left unauthenticated so orchestrators can probe it, but the detail -- which
    camera, and why -- is only returned to a logged-in caller. An anonymous probe
    gets the verdict and a count.
    """
    ok, problems = supervisor.healthy()
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    if await resolve_principal(request) is not None:
        return {"ready": ok, "problems": problems}
    return {"ready": ok, "problem_count": len(problems)}


@router.get("/metrics", dependencies=[Depends(metrics_access)])
async def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/api/config/reload", dependencies=[Depends(require_admin)])
async def reload_config(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    try:
        site = await supervisor.reload_config()
    except Exception as exc:  # noqa: BLE001 - surface validation errors to the operator
        log.exception("config reload failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid config: {exc}"
        ) from exc
    return {"reloaded": True, "cameras": len(site.cameras), "contacts": len(site.contacts)}


@router.get("/api/config", dependencies=[Depends(require_admin)])
async def get_config(
    supervisor: Supervisor = Depends(get_supervisor),
    settings: Settings = Depends(get_settings),
) -> dict:
    """The effective site config. Never includes secrets -- those live in the env."""
    return {
        "site": {"name": supervisor.site.name, "timezone": supervisor.site.timezone},
        "alerting": supervisor.site.alerting.model_dump(mode="json"),
        # public_dict, not model_dump: camera passwords must not leave the process.
        "cameras": [c.public_dict() for c in supervisor.site.cameras],
        "contacts": [c.model_dump(mode="json") for c in supervisor.site.contacts],
        "detection": supervisor.detection_summary(),
        "runtime": {
            "shadow_mode": supervisor.shadow_mode,
            "detector_backend": settings.detector_backend,
            "twilio_configured": settings.twilio_configured(),
            "public_base_url": settings.public_base_url,
        },
    }
