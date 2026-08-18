"""Status, health, metrics and configuration reload."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import Settings
from ..supervisor import Supervisor
from .deps import get_settings, get_supervisor

log = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/api/status")
async def get_status(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    return supervisor.status()


@router.get("/api/stats")
async def get_stats(days: int = 7, supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    return await supervisor.store.stats(days=days)


@router.get("/api/health")
async def health() -> dict:
    """Liveness only: is the process serving? Kept trivial on purpose."""
    return {"status": "ok"}


@router.get("/api/ready")
async def ready(response: Response, supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    """Readiness. A disconnected camera is a real degradation and reports as one."""
    ok, problems = supervisor.healthy()
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ok, "problems": problems}


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/api/config/reload")
async def reload_config(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    try:
        site = await supervisor.reload_config()
    except Exception as exc:  # noqa: BLE001 - surface validation errors to the operator
        log.exception("config reload failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid config: {exc}"
        ) from exc
    return {"reloaded": True, "cameras": len(site.cameras), "contacts": len(site.contacts)}


@router.get("/api/config")
async def get_config(
    supervisor: Supervisor = Depends(get_supervisor),
    settings: Settings = Depends(get_settings),
) -> dict:
    """The effective site config. Never includes secrets -- those live in the env."""
    return {
        "site": {"name": supervisor.site.name, "timezone": supervisor.site.timezone},
        "alerting": supervisor.site.alerting.model_dump(mode="json"),
        "cameras": [c.model_dump(mode="json") for c in supervisor.site.cameras],
        "contacts": [c.model_dump(mode="json") for c in supervisor.site.contacts],
        "runtime": {
            "shadow_mode": settings.shadow_mode,
            "detector_backend": settings.detector_backend,
            "twilio_configured": settings.twilio_configured(),
            "public_base_url": settings.public_base_url,
        },
    }
