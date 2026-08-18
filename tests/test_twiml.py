"""TwiML generation and URL redaction."""

from __future__ import annotations

from xml.etree import ElementTree

from firemex.ingest.sources import _redact
from firemex.notify.twilio_voice import build_alert_twiml


def parse(twiml: str) -> ElementTree.Element:
    return ElementTree.fromstring(twiml)


def test_alert_twiml_is_well_formed_xml():
    root = parse(build_alert_twiml("Fire alert", "https://example.com/ack/inc-1"))
    assert root.tag == "Response"


def test_alert_gathers_a_digit_so_reach_can_be_distinguished_from_voicemail():
    """Without an explicit acknowledgement you cannot tell a contact who answered
    from an answering machine, and a chain that stops on ringing is dangerous."""
    root = parse(build_alert_twiml("Fire alert", "https://example.com/ack/inc-1"))
    gather = root.find("Gather")
    assert gather is not None
    assert gather.get("numDigits") == "1"
    assert gather.get("action") == "https://example.com/ack/inc-1"
    assert gather.get("method") == "POST"


def test_message_is_repeated_inside_the_gather():
    """The first seconds of an unexpected call are routinely missed, and a digit
    pressed during the message must still be accepted."""
    root = parse(build_alert_twiml("Fire alert at Warehouse 3", "https://example.com/ack"))
    says = root.find("Gather").findall("Say")
    assert len(says) == 2
    assert all("Warehouse 3" in (say.text or "") for say in says)


def test_a_prerecorded_clip_replaces_the_spoken_message():
    root = parse(
        build_alert_twiml("ignored", "https://example.com/ack", clip_url="https://cdn/alert.mp3")
    )
    gather = root.find("Gather")
    assert gather.find("Say") is None
    play = gather.find("Play")
    assert play.text == "https://cdn/alert.mp3"
    assert play.get("loop") == "2"


def test_falling_through_the_gather_says_so_and_hangs_up():
    """Escalation is driven by the dispatcher's timeout, so the call must end
    cleanly rather than sit open."""
    root = parse(build_alert_twiml("Fire alert", "https://example.com/ack"))
    tail = list(root)[1:]
    assert tail[0].tag == "Say"
    assert "Escalating" in tail[0].text
    assert tail[-1].tag == "Hangup"


def test_special_characters_are_escaped_not_injected():
    root = parse(
        build_alert_twiml(
            'Fire at "Bay 3" & <Loading>', "https://example.com/ack?a=1&b=2"
        )
    )
    say = root.find("Gather").find("Say")
    assert say.text == 'Fire at "Bay 3" & <Loading>'
    assert root.find("Gather").get("action") == "https://example.com/ack?a=1&b=2"


def test_rtsp_credentials_are_redacted_from_logs():
    """Camera passwords must never reach a log line."""
    redacted = _redact("rtsp://admin:secret123@192.168.1.40:554/Streaming/Channels/101")
    assert "secret123" not in redacted
    assert "admin" not in redacted
    assert "192.168.1.40:554" in redacted


def test_redaction_leaves_credential_free_urls_alone():
    url = "rtsp://192.168.1.40:554/stream"
    assert _redact(url) == url
