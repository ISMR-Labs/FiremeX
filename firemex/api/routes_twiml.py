"""Twilio webhooks: alert TwiML, acknowledgement, and call status.

Every endpoint here is signature-verified. They are necessarily reachable from the
public internet, and an unsigned request must never be able to silence a fire
alert.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Request, Response

from ..config import Settings
from ..notify.twilio_voice import build_alert_twiml
from ..supervisor import Supervisor
from .deps import get_settings, get_supervisor, verify_twilio_signature

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/twiml", tags=["twilio"], dependencies=[Depends(verify_twilio_signature)]
)

XML = "application/xml"


@router.post("/alert/{incident_id}")
async def alert_twiml(
    incident_id: str,
    request: Request,
    supervisor: Supervisor = Depends(get_supervisor),
    settings: Settings = Depends(get_settings),
) -> Response:
    """The spoken alert, plus a gather so the contact can acknowledge."""
    contact_id = request.query_params.get("contact", "unknown")
    incident = await supervisor.store.get_incident(incident_id)
    alerting = supervisor.site.alerting

    if incident is None:
        # A self-test call, or an incident already purged. Say something true.
        message = (
            f"This is a FiremeX test call for {supervisor.site.name}. "
            "Press 1 to confirm you received this."
        )
    else:
        labels = (incident.get("labels") or "smoke").replace(",", " and ")
        message = alerting.voice_template.format(
            site=supervisor.site.name,
            camera=incident.get("camera_name") or incident.get("camera_id", "unknown camera"),
            camera_id=incident.get("camera_id", ""),
            location=incident.get("location") or "location not set",
            labels=labels.capitalize(),
            severity=incident.get("severity", "warning"),
            link="",
        )

    base = settings.public_base_url.rstrip("/")
    ack_url = f"{base}/twiml/ack/{incident_id}?contact={contact_id}"
    twiml = build_alert_twiml(
        message=message, ack_url=ack_url, clip_url=alerting.voice_clip_url
    )
    return Response(content=twiml, media_type=XML)


@router.post("/ack/{incident_id}")
async def acknowledge(
    incident_id: str,
    request: Request,
    Digits: str = Form(default=""),
    supervisor: Supervisor = Depends(get_supervisor),
) -> Response:
    """Handle the gathered digit. Pressing 1 stops the escalation chain."""
    contact_id = request.query_params.get("contact", "unknown")
    if Digits.strip() == "1":
        await supervisor.acknowledge_incident(incident_id, contact_id)
        spoken = (
            "Acknowledged. Escalation stopped. "
            "Please check the camera and call your fire service if needed."
        )
    else:
        log.info(
            "incident %s: contact %s pressed %r, not acknowledging",
            incident_id,
            contact_id,
            Digits,
        )
        spoken = "That was not a valid response. Escalating to the next contact."
    return Response(
        content=(
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Say>{spoken}</Say><Hangup/></Response>"
        ),
        media_type=XML,
    )


@router.post("/status/{incident_id}")
async def call_status(
    incident_id: str,
    request: Request,
    CallStatus: str = Form(default=""),
    CallSid: str = Form(default=""),
) -> Response:
    """Twilio call-progress callback.

    Logged rather than acted upon: call progress says the phone was answered, not
    that a human understood. Only the gathered digit stops escalation.
    """
    contact_id = request.query_params.get("contact", "unknown")
    log.info(
        "call status incident=%s contact=%s sid=%s status=%s",
        incident_id,
        contact_id,
        CallSid,
        CallStatus,
    )
    return Response(status_code=204)
