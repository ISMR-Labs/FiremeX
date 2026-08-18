"""Twilio voice and SMS channels.

Design notes that matter operationally:

* The call plays an alert and then **gathers a digit**. Without an explicit
  acknowledgement you cannot distinguish "contact reached" from "call went to
  voicemail", and a chain that stops on ringing is worse than no chain.
* A pre-recorded clip (``voice_clip_url``) is clearer under stress than TTS and
  removes speech-synthesis latency from the alert path. ``<Say>`` is the fallback.
* SMS carries the snapshot link; the call gets attention. Both are sent.
* Twilio does **not** provide general-purpose emergency calling, and automated
  false emergency calls are an offence in most jurisdictions. These channels call
  the site's own responders. Do not point them at 911/112/119.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote

from .. import metrics
from ..config import ContactConfig, Settings
from .base import AlertContext, AlertResult, Outcome

log = logging.getLogger(__name__)


def build_alert_twiml(
    message: str,
    ack_url: str,
    clip_url: str | None = None,
    repeat: int = 2,
    voice: str = "Polly.Joanna",
    language: str = "en-US",
) -> str:
    """TwiML for the alert call: announce, then gather one digit.

    The message repeats because the first seconds of an unexpected call are
    routinely missed, and ``<Gather>`` wraps the announcement so pressing 1 during
    the message is accepted rather than ignored until the end.
    """
    if clip_url:
        body = f'<Play loop="{repeat}">{_escape(clip_url)}</Play>'
    else:
        body = "".join(
            f'<Say voice="{voice}" language="{language}">{_escape(message)}</Say>'
            for _ in range(repeat)
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Gather numDigits="1" timeout="8" action="{_escape(ack_url)}" method="POST">'
        f"{body}"
        "</Gather>"
        # Reached only if no digit was pressed: say so plainly and hang up, so the
        # dispatcher's timeout is the single source of truth for escalation.
        f'<Say voice="{voice}" language="{language}">No acknowledgement received. '
        "Escalating to the next contact.</Say>"
        "<Hangup/>"
        "</Response>"
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class TwilioVoiceChannel:
    name = "call"

    def __init__(self, settings: Settings, client=None) -> None:
        self.settings = settings
        self._client = client
        if client is None and settings.twilio_configured():
            from twilio.rest import Client

            self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def send(
        self, context: AlertContext, contact: ContactConfig, message: str
    ) -> AlertResult:
        if not self.available:
            log.error("cannot place call: Twilio is not configured")
            metrics.ALERTS_SENT.labels(channel=self.name, outcome="unconfigured").inc()
            return AlertResult(
                channel=self.name,
                contact_id=contact.id,
                outcome=Outcome.FAILED,
                error="twilio_not_configured",
            )

        base = self.settings.public_base_url.rstrip("/")
        twiml_url = (
            f"{base}/twiml/alert/{quote(context.incident_id)}"
            f"?contact={quote(contact.id)}"
        )
        status_url = f"{base}/twiml/status/{quote(context.incident_id)}?contact={quote(contact.id)}"
        try:
            call = await asyncio.to_thread(
                self._client.calls.create,
                to=contact.phone,
                from_=self.settings.twilio_from_number,
                url=twiml_url,
                method="POST",
                status_callback=status_url,
                status_callback_event=["initiated", "answered", "completed"],
                status_callback_method="POST",
                # Long enough for voicemail to pick up and the gather to fail
                # honestly, short enough not to hold the chain open.
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors must not stop the chain
            log.exception("call to %s (%s) failed", contact.name, contact.id)
            metrics.ALERTS_SENT.labels(channel=self.name, outcome="failed").inc()
            return AlertResult(
                channel=self.name, contact_id=contact.id, outcome=Outcome.FAILED, error=str(exc)
            )

        log.info("call queued to %s (%s) sid=%s", contact.name, contact.phone, call.sid)
        metrics.ALERTS_SENT.labels(channel=self.name, outcome="queued").inc()
        return AlertResult(
            channel=self.name,
            contact_id=contact.id,
            outcome=Outcome.QUEUED,
            provider_id=call.sid,
        )


class TwilioSmsChannel:
    name = "sms"

    def __init__(self, settings: Settings, client=None) -> None:
        self.settings = settings
        self._client = client
        if client is None and settings.twilio_configured():
            from twilio.rest import Client

            self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def send(
        self, context: AlertContext, contact: ContactConfig, message: str
    ) -> AlertResult:
        if not self.available:
            metrics.ALERTS_SENT.labels(channel=self.name, outcome="unconfigured").inc()
            return AlertResult(
                channel=self.name,
                contact_id=contact.id,
                outcome=Outcome.FAILED,
                error="twilio_not_configured",
            )
        try:
            sms = await asyncio.to_thread(
                self._client.messages.create,
                to=contact.phone,
                from_=self.settings.twilio_from_number,
                body=message[:1500],
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("sms to %s (%s) failed", contact.name, contact.id)
            metrics.ALERTS_SENT.labels(channel=self.name, outcome="failed").inc()
            return AlertResult(
                channel=self.name, contact_id=contact.id, outcome=Outcome.FAILED, error=str(exc)
            )
        metrics.ALERTS_SENT.labels(channel=self.name, outcome="queued").inc()
        return AlertResult(
            channel=self.name, contact_id=contact.id, outcome=Outcome.QUEUED, provider_id=sms.sid
        )
