"""Camera and contact management.

Edits are written back to the site YAML and applied by reloading the supervisor,
so the file on disk stays the single source of truth. A dashboard change and a
hand-edit of ``config.yaml`` therefore cannot drift apart.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..config import CameraConfig, ContactConfig, SiteConfig, dump_site_config
from ..supervisor import Supervisor
from .deps import get_supervisor

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["configuration"])


class SiteUpdate(BaseModel):
    name: str | None = None
    timezone: str | None = None


async def _persist(supervisor: Supervisor, site: SiteConfig) -> SiteConfig:
    """Validate, write, then reload. Validation happens by reconstructing the model
    so a bad reference can never reach disk."""
    try:
        validated = SiteConfig.model_validate(site.model_dump(mode="python"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid configuration: {exc}"
        ) from exc
    dump_site_config(validated, supervisor.settings.config_path)
    return await supervisor.reload_config()


@router.get("/cameras")
async def list_cameras(supervisor: Supervisor = Depends(get_supervisor)) -> list[dict]:
    return [camera.model_dump(mode="json") for camera in supervisor.site.cameras]


@router.post("/cameras", status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera: CameraConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if supervisor.site.camera(camera.id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"camera {camera.id!r} already exists"
        )
    site = supervisor.site.model_copy(deep=True)
    site.cameras.append(camera)
    await _persist(supervisor, site)
    return camera.model_dump(mode="json")


@router.put("/cameras/{camera_id}")
async def update_camera(
    camera_id: str, camera: CameraConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if camera.id != camera_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="camera id in body must match the path"
        )
    site = supervisor.site.model_copy(deep=True)
    index = next((i for i, c in enumerate(site.cameras) if c.id == camera_id), None)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    site.cameras[index] = camera
    await _persist(supervisor, site)
    return camera.model_dump(mode="json")


@router.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: str, supervisor: Supervisor = Depends(get_supervisor)) -> None:
    site = supervisor.site.model_copy(deep=True)
    remaining = [c for c in site.cameras if c.id != camera_id]
    if len(remaining) == len(site.cameras):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    site.cameras = remaining
    await _persist(supervisor, site)


@router.get("/contacts")
async def list_contacts(supervisor: Supervisor = Depends(get_supervisor)) -> list[dict]:
    return [contact.model_dump(mode="json") for contact in supervisor.site.contacts]


@router.post("/contacts", status_code=status.HTTP_201_CREATED)
async def create_contact(
    contact: ContactConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if supervisor.site.contact(contact.id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"contact {contact.id!r} already exists"
        )
    site = supervisor.site.model_copy(deep=True)
    site.contacts.append(contact)
    await _persist(supervisor, site)
    return contact.model_dump(mode="json")


@router.put("/contacts/{contact_id}")
async def update_contact(
    contact_id: str, contact: ContactConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if contact.id != contact_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="contact id in body must match the path"
        )
    site = supervisor.site.model_copy(deep=True)
    index = next((i for i, c in enumerate(site.contacts) if c.id == contact_id), None)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="contact not found")
    site.contacts[index] = contact
    await _persist(supervisor, site)
    return contact.model_dump(mode="json")


@router.delete("/contacts/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_contact(contact_id: str, supervisor: Supervisor = Depends(get_supervisor)) -> None:
    site = supervisor.site.model_copy(deep=True)
    remaining = [c for c in site.contacts if c.id != contact_id]
    if len(remaining) == len(site.contacts):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="contact not found")
    # Drop dangling references in the same write so the config stays valid.
    site.contacts = remaining
    for camera in site.cameras:
        camera.contacts = [ref for ref in camera.contacts if ref != contact_id]
    site.alerting.default_contacts = [
        ref for ref in site.alerting.default_contacts if ref != contact_id
    ]
    await _persist(supervisor, site)


@router.patch("/site")
async def update_site(
    update: SiteUpdate, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    site = supervisor.site.model_copy(deep=True)
    if update.name is not None:
        site.name = update.name
    if update.timezone is not None:
        site.timezone = update.timezone
    result = await _persist(supervisor, site)
    return {"name": result.name, "timezone": result.timezone}


@router.post("/contacts/{contact_id}/test-call")
async def test_call(contact_id: str, supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    """Place a real test call.

    Untested alerting is broken alerting: this is the button an operator presses
    after changing a phone number, and the monthly self-test drives the same path.
    """
    contact = supervisor.site.contact(contact_id)
    if contact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="contact not found")
    if supervisor.dispatcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="dispatcher not ready"
        )
    channel = supervisor.dispatcher.channels.get("call")
    if channel is None or not getattr(channel, "available", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Twilio is not configured; cannot place a test call",
        )

    from ..notify.base import AlertContext

    context = AlertContext(
        incident_id=f"selftest-{contact_id}",
        camera_id="selftest",
        camera_name="Self test",
        location="test",
        site_name=supervisor.site.name,
        labels="Test alert",
        severity="warning",
    )
    message = (
        f"This is a FiremeX test call for {supervisor.site.name}. "
        "No fire has been detected. Press 1 to confirm you received this."
    )
    result = await channel.send(context, contact, message)
    await supervisor.store.record_self_test(
        kind="call", contact_id=contact_id, outcome=result.outcome.value, detail=result.error
    )
    return {
        "contact_id": contact_id,
        "outcome": result.outcome.value,
        "provider_id": result.provider_id,
        "error": result.error,
    }
