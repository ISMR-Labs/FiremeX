"""Escalation behaviour.

The properties under test are the ones that decide whether a real fire reaches a
human: stop on acknowledgement, escalate on silence, one call sequence per
incident, and never call at all in shadow mode.
"""

from __future__ import annotations

import asyncio

import pytest

from firemex.detect.base import BBox
from firemex.incident.engine import Incident, Severity
from firemex.notify.base import AckBus, AlertContext, AlertResult, CooldownStore, Outcome
from firemex.notify.dispatcher import AlertDispatcher, RunStatus


def make_incident(camera_id="cam-1", incident_id="inc-1", severity=Severity.CRITICAL):
    return Incident(
        id=incident_id,
        camera_id=camera_id,
        opened_monotonic=100.0,
        opened_wall=1_700_000_000.0,
        severity=severity,
        labels={"fire"},
        peak_confidence=0.91,
        envelope=BBox(0.4, 0.4, 0.6, 0.6),
        frames_confirmed=6,
        last_hit_monotonic=100.0,
    )


class RecordingChannel:
    """Captures calls and optionally acknowledges on a chosen attempt."""

    def __init__(self, name="call", ack_bus=None, ack_on=None, fail_for=()):
        self.name = name
        self.sent: list[tuple[str, str]] = []
        self._ack_bus = ack_bus
        self._ack_on = ack_on
        self._fail_for = set(fail_for)

    async def send(self, context: AlertContext, contact, message: str) -> AlertResult:
        self.sent.append((contact.id, message))
        if contact.id in self._fail_for:
            return AlertResult(self.name, contact.id, Outcome.FAILED, error="simulated failure")
        if self._ack_bus is not None and self._ack_on == len(self.sent):
            # The contact presses 1 while the call is still in flight.
            self._ack_bus.acknowledge(context.incident_id, contact.id)
        return AlertResult(
            self.name, contact.id, Outcome.QUEUED, provider_id=f"sid-{len(self.sent)}"
        )

    @property
    def contacts_called(self) -> list[str]:
        return [contact_id for contact_id, _ in self.sent]


async def instant_sleep(_seconds: float) -> None:
    """Collapse the confirm-delay and ack windows so tests run at full speed."""
    await asyncio.sleep(0)


def build(site, channels, *, shadow=False, ack_bus=None, sleep=instant_sleep, cooldowns=None):
    return AlertDispatcher(
        site=site,
        channels=channels,
        ack_bus=ack_bus,
        cooldowns=cooldowns or CooldownStore(),
        shadow_mode=shadow,
        sleep=sleep,
        dashboard_url="http://testserver",
    )


# ---- shadow mode ---------------------------------------------------------


async def test_shadow_mode_places_no_calls(site, camera, contact):
    subject = camera()
    channel = RecordingChannel()
    configured = site(cameras=[subject], contacts=[contact()], default_contacts=["primary"])
    dispatcher = build(configured, {"call": channel}, shadow=True)

    run = await dispatcher.run(make_incident(), subject)

    assert run.status is RunStatus.SHADOW
    assert channel.sent == [], "shadow mode must never place a call"


async def test_shadow_mode_still_fires_webhooks(site, camera, contact):
    """Webhooks are how a site wires FiremeX into an existing BMS, and carry no
    risk of a false phone call, so they run even in shadow mode."""
    delivered = []

    class Webhook:
        async def broadcast(self, payload):
            delivered.append(payload)
            return []

    subject = camera()
    configured = site(cameras=[subject], contacts=[contact()], default_contacts=["primary"])
    dispatcher = build(configured, {}, shadow=True)
    dispatcher.webhook = Webhook()

    await dispatcher.run(make_incident(), subject)

    assert len(delivered) == 1
    assert delivered[0]["event"] == "incident.confirmed"
    assert delivered[0]["shadow_mode"] is True


# ---- escalation ----------------------------------------------------------


async def test_acknowledgement_stops_the_chain(site, camera, contact):
    subject = camera()
    ack_bus = AckBus()
    channel = RecordingChannel(ack_bus=ack_bus, ack_on=1)
    configured = site(
        cameras=[subject],
        contacts=[contact("primary"), contact("secondary", phone="+10000000002")],
        default_contacts=["primary", "secondary"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=ack_bus)

    run = await dispatcher.run(make_incident(), subject)

    assert run.status is RunStatus.ACKNOWLEDGED
    assert run.acknowledged_by == "primary"
    assert channel.contacts_called == ["primary"], "must not call anyone after an ack"


async def test_silence_escalates_through_the_whole_chain(site, camera, contact):
    subject = camera()
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[
            contact("primary"),
            contact("secondary", phone="+10000000002"),
            contact("owner", phone="+10000000003"),
        ],
        default_contacts=["primary", "secondary", "owner"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus())

    run = await dispatcher.run(make_incident(), subject)

    assert channel.contacts_called == ["primary", "secondary", "owner"]
    assert run.status is RunStatus.EXHAUSTED


async def test_retries_are_attempted_before_escalating(site, camera, contact):
    subject = camera()
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[contact("primary", retries=2), contact("secondary", phone="+10000000002")],
        default_contacts=["primary", "secondary"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus())

    await dispatcher.run(make_incident(), subject)

    # 3 attempts for primary (retries=2), then 1 for secondary.
    assert channel.contacts_called == ["primary", "primary", "primary", "secondary"]


async def test_ack_on_a_later_retry_stops_the_chain(site, camera, contact):
    subject = camera()
    ack_bus = AckBus()
    channel = RecordingChannel(ack_bus=ack_bus, ack_on=3)
    configured = site(
        cameras=[subject],
        contacts=[contact("primary", retries=3), contact("secondary", phone="+10000000002")],
        default_contacts=["primary", "secondary"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=ack_bus)

    run = await dispatcher.run(make_incident(), subject)

    assert run.status is RunStatus.ACKNOWLEDGED
    assert channel.contacts_called == ["primary", "primary", "primary"]


async def test_provider_failure_does_not_stall_the_chain(site, camera, contact):
    """A dead Twilio must not consume the acknowledgement window."""
    subject = camera()
    channel = RecordingChannel(fail_for={"primary"})
    configured = site(
        cameras=[subject],
        contacts=[contact("primary", retries=1), contact("secondary", phone="+10000000002")],
        default_contacts=["primary", "secondary"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus())

    run = await dispatcher.run(make_incident(), subject)

    assert channel.contacts_called == ["primary", "primary", "secondary"]
    failures = [a for a in run.attempts if a.outcome is Outcome.FAILED]
    assert len(failures) == 2


async def test_sms_is_sent_once_per_contact(site, camera, contact):
    subject = camera()
    call = RecordingChannel("call")
    sms = RecordingChannel("sms")
    configured = site(
        cameras=[subject],
        contacts=[contact("primary", retries=2, channels=["call", "sms"])],
        default_contacts=["primary"],
    )
    dispatcher = build(configured, {"call": call, "sms": sms}, ack_bus=AckBus())

    await dispatcher.run(make_incident(), subject)

    assert len(call.sent) == 3
    assert len(sms.sent) == 1, "repeat texts add noise without adding reach"


async def test_camera_contacts_take_precedence_over_site_default(site, camera, contact):
    subject = camera(contacts=["owner"])
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[contact("primary"), contact("owner", phone="+10000000003")],
        default_contacts=["primary"],
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus())

    await dispatcher.run(make_incident(), subject)

    assert channel.contacts_called == ["owner"]


async def test_no_contacts_is_reported_not_swallowed(site, camera):
    subject = camera()
    channel = RecordingChannel()
    configured = site(cameras=[subject], contacts=[])
    dispatcher = build(configured, {"call": channel})

    run = await dispatcher.run(make_incident(), subject)

    assert run.status is RunStatus.NO_CONTACTS
    assert channel.sent == []


# ---- de-duplication ------------------------------------------------------


async def test_cooldown_suppresses_a_second_incident_on_the_same_camera(site, camera, contact):
    """A ten-minute fire must produce one call sequence, not two hundred."""
    subject = camera()
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[contact()],
        default_contacts=["primary"],
        cooldown_minutes=10,
    )
    cooldowns = CooldownStore()
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus(), cooldowns=cooldowns)

    first = await dispatcher.run(make_incident(incident_id="inc-1"), subject)
    second = await dispatcher.run(make_incident(incident_id="inc-2"), subject)

    assert first.status is RunStatus.EXHAUSTED
    assert second.status is RunStatus.SUPPRESSED
    assert channel.contacts_called == ["primary"]


async def test_cooldown_is_per_camera(site, camera, contact):
    cam_a, cam_b = camera(id="cam-a"), camera(id="cam-b")
    channel = RecordingChannel()
    configured = site(
        cameras=[cam_a, cam_b], contacts=[contact()], default_contacts=["primary"]
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus())

    await dispatcher.run(make_incident(camera_id="cam-a", incident_id="a1"), cam_a)
    run_b = await dispatcher.run(make_incident(camera_id="cam-b", incident_id="b1"), cam_b)

    assert run_b.status is RunStatus.EXHAUSTED
    assert channel.contacts_called == ["primary", "primary"]


# ---- the cancel window ---------------------------------------------------


async def test_operator_can_cancel_within_the_confirm_delay(site, camera, contact):
    """The grace period exists so a human can stop a false alarm before any phone
    rings. This test asserts that no call is placed when they do."""
    subject = camera()
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[contact()],
        default_contacts=["primary"],
        confirm_delay_seconds=30.0,
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus(), sleep=asyncio.sleep)

    incident = make_incident()
    task = dispatcher.dispatch(incident, subject)
    # Let the run reach the confirm-delay sleep.
    await asyncio.sleep(0.05)
    assert dispatcher.cancel(incident.id, "operator dismissed") is True

    with pytest.raises(asyncio.CancelledError):
        await task

    assert channel.sent == [], "cancelling inside the grace period must place no call"
    assert dispatcher.runs[incident.id].status is RunStatus.CANCELLED


async def test_cancel_after_completion_reports_nothing_to_cancel(site, camera, contact):
    subject = camera()
    configured = site(cameras=[subject], contacts=[contact()], default_contacts=["primary"])
    dispatcher = build(configured, {"call": RecordingChannel()}, ack_bus=AckBus())

    incident = make_incident()
    await dispatcher.dispatch(incident, subject)
    assert dispatcher.cancel(incident.id) is False


async def test_dispatch_is_idempotent_per_incident(site, camera, contact):
    subject = camera()
    channel = RecordingChannel()
    configured = site(
        cameras=[subject],
        contacts=[contact()],
        default_contacts=["primary"],
        confirm_delay_seconds=5.0,
    )
    dispatcher = build(configured, {"call": channel}, ack_bus=AckBus(), sleep=asyncio.sleep)

    incident = make_incident()
    first = dispatcher.dispatch(incident, subject)
    second = dispatcher.dispatch(incident, subject)
    assert first is second, "one incident must not start two escalation runs"

    dispatcher.cancel(incident.id)
    with pytest.raises(asyncio.CancelledError):
        await first


# ---- the acknowledgement bus --------------------------------------------


async def test_ack_bus_records_an_early_acknowledgement():
    """A responder can press 1 before the loop starts waiting; that must count."""
    bus = AckBus()
    assert bus.acknowledge("inc-1", "primary") is True
    assert bus.acknowledge("inc-1", "secondary") is False, "first ack wins"
    assert bus.acknowledged_by("inc-1") == "primary"
    assert await bus.wait("inc-1", timeout=0.01) is True


async def test_ack_bus_times_out_when_nobody_answers():
    bus = AckBus()
    assert await bus.wait("inc-1", timeout=0.01) is False
    assert not bus.is_acknowledged("inc-1")


async def test_ack_bus_wakes_a_waiter():
    bus = AckBus()

    async def press_one():
        await asyncio.sleep(0.01)
        bus.acknowledge("inc-1", "primary")

    asyncio.create_task(press_one())
    assert await bus.wait("inc-1", timeout=1.0) is True


async def test_ack_bus_forget_clears_state():
    bus = AckBus()
    bus.acknowledge("inc-1", "primary")
    bus.forget("inc-1")
    assert not bus.is_acknowledged("inc-1")


# ---- cooldown store ------------------------------------------------------


async def test_cooldown_store_expires():
    store = CooldownStore()
    assert await store.should_alert("cam-1", now=0.0, cooldown_seconds=60.0) is True
    assert await store.should_alert("cam-1", now=30.0, cooldown_seconds=60.0) is False
    assert await store.should_alert("cam-1", now=61.0, cooldown_seconds=60.0) is True


async def test_cooldown_store_clear():
    store = CooldownStore()
    await store.should_alert("cam-1", now=0.0, cooldown_seconds=60.0)
    await store.clear("cam-1")
    assert await store.should_alert("cam-1", now=1.0, cooldown_seconds=60.0) is True


# ---- message rendering ---------------------------------------------------


def test_alert_context_renders_the_voice_template(site):
    configured = site()
    context = AlertContext(
        incident_id="inc-1",
        camera_id="loading-bay",
        camera_name="Loading Bay",
        location="Ground floor, east",
        site_name="Warehouse 3",
        labels="Fire and smoke",
        severity="critical",
    )
    spoken = context.render(configured.alerting.voice_template)
    assert "Warehouse 3" in spoken
    assert "Loading Bay" in spoken
    assert "Ground floor, east" in spoken
    assert "Press 1" in spoken


def test_alert_context_handles_a_missing_location(site):
    context = AlertContext(
        incident_id="inc-1",
        camera_id="cam-1",
        camera_name="Cam 1",
        location="",
        site_name="Site",
        labels="Smoke",
        severity="warning",
    )
    assert "location not set" in context.render(site().alerting.voice_template)
