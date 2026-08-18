"""Incident history, evidence, and the false-positive review queue."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..supervisor import Supervisor
from .deps import get_supervisor

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/incidents", tags=["incidents"])


class ReviewRequest(BaseModel):
    #: ``false_positive`` is the label that feeds fine-tuning. Recording it is the
    #: cheapest thing an operator can do to make the detector better.
    verdict: Literal["real", "false_positive", "drill", "unclear"]
    note: str | None = None


class CancelRequest(BaseModel):
    reason: str = "operator cancelled from dashboard"


@router.get("")
async def list_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    camera_id: str | None = None,
    review: str | None = None,
    unreviewed_only: bool = False,
    supervisor: Supervisor = Depends(get_supervisor),
) -> list[dict]:
    return await supervisor.store.list_incidents(
        limit=limit,
        offset=offset,
        camera_id=camera_id,
        review=review,
        unreviewed_only=unreviewed_only,
    )


@router.get("/{incident_id}")
async def get_incident(
    incident_id: str, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    incident = await supervisor.store.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="incident not found")
    run = supervisor.dispatcher.runs.get(incident_id) if supervisor.dispatcher else None
    incident["dispatch"] = run.as_dict() if run else None
    return incident


@router.post("/{incident_id}/cancel")
async def cancel_incident(
    incident_id: str,
    body: CancelRequest | None = None,
    supervisor: Supervisor = Depends(get_supervisor),
) -> dict:
    """Stop escalation and mark the incident a false positive."""
    reason = (body or CancelRequest()).reason
    cancelled = await supervisor.cancel_incident(incident_id, reason)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no such incident to cancel"
        )
    return {"incident_id": incident_id, "cancelled": True, "reason": reason}


@router.post("/{incident_id}/review")
async def review_incident(
    incident_id: str, body: ReviewRequest, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    updated = await supervisor.store.update_incident(
        incident_id, review=body.verdict, review_note=body.note
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="incident not found")
    return updated


@router.get("/{incident_id}/snapshot")
async def get_snapshot(incident_id: str, supervisor: Supervisor = Depends(get_supervisor)):
    snapshot_path, _ = await supervisor.store.incident_paths(incident_id)
    return _serve(snapshot_path, "image/jpeg", "snapshot")


@router.get("/{incident_id}/clip")
async def get_clip(incident_id: str, supervisor: Supervisor = Depends(get_supervisor)):
    _, clip_path = await supervisor.store.incident_paths(incident_id)
    return _serve(clip_path, "video/mp4", "clip")


def _serve(path: str | None, media_type: str, what: str) -> FileResponse:
    if not path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no {what} recorded")
    file_path = Path(path)
    if not file_path.exists():
        log.warning("%s missing from disk: %s", what, file_path)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} file missing")
    return FileResponse(file_path, media_type=media_type)
