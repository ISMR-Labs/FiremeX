"""Camera and contact management.

Edits are written back to the site YAML and applied by reloading the supervisor,
so the file on disk stays the single source of truth. A dashboard change and a
hand-edit of ``config.yaml`` therefore cannot drift apart.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..config import (
    AlertingConfig,
    CameraConfig,
    ContactConfig,
    DetectionConfig,
    SiteConfig,
    dump_site_config,
)
from ..supervisor import Supervisor
from .deps import get_supervisor, require_admin

log = logging.getLogger(__name__)
# Configuration is admin-only. Operators run incidents; they do not retune the
# detector or edit who gets called.
router = APIRouter(prefix="/api", tags=["configuration"], dependencies=[Depends(require_admin)])


class SiteUpdate(BaseModel):
    name: str | None = None
    timezone: str | None = None


class CameraUpsert(CameraConfig):
    """Camera payload from the UI.

    ``password`` is write-only: omitting it on an update keeps the stored one, so
    editing a camera's frame rate does not require retyping its credentials. An
    explicit empty string clears it.
    """


async def _persist(supervisor: Supervisor, site: SiteConfig) -> SiteConfig:
    """Validate, write, then reload. Validation happens by reconstructing the model
    so a bad reference can never reach disk."""
    try:
        validated = SiteConfig.model_validate(site.model_dump(mode="python"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid configuration: {exc}"
        ) from exc
    try:
        dump_site_config(validated, supervisor.settings.config_path)
    except OSError as exc:
        # Usually a read-only mount or a permissions mistake. Say so plainly:
        # an ASGI traceback tells the operator nothing they can act on.
        path = supervisor.settings.config_path
        log.error("cannot write the site config to %s: %s", path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"could not save the configuration to {path}: {exc.strerror or exc}. "
                "The dashboard writes this file, so it must be writable by the server "
                "(in Docker, check that the config.yaml mount is not read-only)."
            ),
        ) from exc
    return await supervisor.reload_config()


@router.get("/cameras")
async def list_cameras(supervisor: Supervisor = Depends(get_supervisor)) -> list[dict]:
    return [camera.public_dict() for camera in supervisor.site.cameras]


@router.post("/cameras", status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera: CameraUpsert, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if supervisor.site.camera(camera.id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"camera {camera.id!r} already exists"
        )
    site = supervisor.site.model_copy(deep=True)
    created = CameraConfig.model_validate(camera.model_dump())
    site.cameras.append(created)
    await _persist(supervisor, site)
    log.info("camera %s created", created.id)
    return created.public_dict()


@router.put("/cameras/{camera_id}")
async def update_camera(
    camera_id: str, camera: CameraUpsert, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    if camera.id != camera_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="camera id in body must match the path"
        )
    site = supervisor.site.model_copy(deep=True)
    index = next((i for i, c in enumerate(site.cameras) if c.id == camera_id), None)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")

    payload = camera.model_dump()
    if payload.get("password") is None:
        # Not supplied: keep whatever is stored, so editing fps does not require
        # retyping the camera password.
        payload["password"] = site.cameras[index].password
    elif payload["password"] == "":
        payload["password"] = None
    updated = CameraConfig.model_validate(payload)
    site.cameras[index] = updated
    await _persist(supervisor, site)
    log.info("camera %s updated", camera_id)
    return updated.public_dict()


@router.post("/cameras/{camera_id}/test")
async def test_camera(camera_id: str, supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    """Open the stream once and report what came back.

    The button an operator presses after typing an RTSP URL, so a typo is caught
    at configuration time instead of discovered during a fire.
    """
    camera = supervisor.site.camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")

    import asyncio

    from ..auth import redact_url
    from ..ingest.sources import RtspSource, StreamError

    def probe() -> dict:
        source = RtspSource(camera.detect_url)
        try:
            source.open()
            image = source.read()
        except (StreamError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            source.close()
        if image is None:
            return {"ok": False, "error": "stream opened but delivered no frame"}
        return {"ok": True, "width": int(image.shape[1]), "height": int(image.shape[0])}

    result = await asyncio.to_thread(probe)
    result["url"] = redact_url(camera.detect_url)
    return result


@router.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: str, supervisor: Supervisor = Depends(get_supervisor)) -> None:
    site = supervisor.site.model_copy(deep=True)
    remaining = [c for c in site.cameras if c.id != camera_id]
    if len(remaining) == len(site.cameras):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    site.cameras = remaining
    await _persist(supervisor, site)


@router.get("/alerting")
async def get_alerting(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    """Notification settings, including the spoken and texted message templates."""
    alerting = supervisor.site.alerting.model_dump(mode="json")
    alerting["placeholders"] = {
        "site": "Site name",
        "camera": "Camera name",
        "camera_id": "Camera id",
        "location": "Camera location",
        "labels": "What was detected, e.g. 'Fire and smoke'",
        "severity": "warning or critical",
        "link": "Snapshot or dashboard link (SMS only)",
    }
    alerting["shadow_mode"] = supervisor.shadow_mode
    return alerting


@router.put("/alerting")
async def update_alerting(
    body: AlertingConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    """Replace the notification settings.

    Templates are rendered against a dummy context first: a typo like ``{camara}``
    must fail here, not at the moment a phone should have rung.
    """
    from ..notify.base import AlertContext

    probe = AlertContext(
        incident_id="preview",
        camera_id="cam",
        camera_name="Camera",
        location="Location",
        site_name=supervisor.site.name,
        labels="Fire and smoke",
        severity="critical",
    )
    for field, template in (
        ("voice_template", body.voice_template),
        ("sms_template", body.sms_template),
    ):
        try:
            probe.render(template)
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field} has an unknown placeholder: {exc}",
            ) from exc

    site = supervisor.site.model_copy(deep=True)
    site.alerting = body
    await _persist(supervisor, site)
    log.info("alerting settings updated")
    return site.alerting.model_dump(mode="json")


@router.post("/alerting/preview")
async def preview_alerting(
    body: AlertingConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    """Render the templates without saving, so the UI can show the exact wording."""
    from ..notify.base import AlertContext

    camera = supervisor.site.cameras[0] if supervisor.site.cameras else None
    context = AlertContext(
        incident_id="preview",
        camera_id=camera.id if camera else "loading-bay",
        camera_name=camera.name if camera else "Loading Bay",
        location=(camera.location if camera else "") or "Ground floor, east",
        site_name=supervisor.site.name,
        labels="Fire and smoke",
        severity="critical",
        snapshot_url="https://example/snapshot.jpg",
    )
    try:
        return {
            "voice": context.render(body.voice_template),
            "sms": context.render(body.sms_template),
        }
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown placeholder: {exc}"
        ) from exc


@router.get("/detection")
async def get_detection(supervisor: Supervisor = Depends(get_supervisor)) -> dict:
    return supervisor.detection_summary()


@router.put("/detection")
async def update_detection(
    body: DetectionConfig, supervisor: Supervisor = Depends(get_supervisor)
) -> dict:
    """Change model settings.

    Rebuilding the detector pauses detection for a moment, so the response says
    whether that happened. A model path that fails to load is rolled back to the
    previous working detector rather than leaving the site blind.
    """
    site = supervisor.site.model_copy(deep=True)
    site.detection = body
    before = supervisor._detector_fingerprint()
    try:
        await _persist(supervisor, site)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad model path lands here
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"could not apply detection settings: {exc}",
        ) from exc
    summary = supervisor.detection_summary()
    summary["detector_rebuilt"] = before != supervisor._detector_fingerprint()
    return summary


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
