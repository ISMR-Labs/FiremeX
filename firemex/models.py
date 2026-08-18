"""Database schema.

Incidents are the audit trail. When someone asks "why did it call at 3am" or, far
worse, "why did it not call", the answer has to be reconstructable from here: the
detections that confirmed it, the assessment numbers, every call placed, and who
acknowledged.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    camera_name: Mapped[str] = mapped_column(String(200), default="")
    location: Mapped[str] = mapped_column(String(200), default="")
    severity: Mapped[str] = mapped_column(String(16), default="warning")
    labels: Mapped[str] = mapped_column(String(64), default="")
    peak_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    growth_ratio: Mapped[float] = mapped_column(Float, default=1.0)
    frames_confirmed: Mapped[int] = mapped_column(Integer, default=0)
    envelope: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Recorded but not alerted -- the site was running in shadow mode.
    shadow_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    #: Escalation outcome, from notify.dispatcher.RunStatus.
    alert_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    acknowledged_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Operator verdict, set from the dashboard. This is the false-positive review
    #: queue that feeds fine-tuning, so it is a first-class column.
    review: Mapped[str | None] = mapped_column(String(24), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    detections: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    assessment: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    alerts: Mapped[list[AlertRecord]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_incidents_camera_opened", "camera_id", "opened_at"),)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "location": self.location,
            "severity": self.severity,
            "labels": self.labels,
            "peak_confidence": self.peak_confidence,
            "growth_ratio": self.growth_ratio,
            "frames_confirmed": self.frames_confirmed,
            "envelope": self.envelope,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "shadow_mode": self.shadow_mode,
            "alert_status": self.alert_status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "review": self.review,
            "review_note": self.review_note,
            "has_snapshot": bool(self.snapshot_path),
            "has_clip": bool(self.clip_path),
            "detections": self.detections or [],
            "assessment": self.assessment,
            "alerts": [alert.as_dict() for alert in self.alerts],
        }


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("incidents.id", ondelete="CASCADE"), index=True
    )
    channel: Mapped[str] = mapped_column(String(16))
    contact_id: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    incident: Mapped[IncidentRecord] = relationship(back_populates="alerts")

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "contact_id": self.contact_id,
            "outcome": self.outcome,
            "provider_id": self.provider_id,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SelfTestRecord(Base):
    """Alerting self-tests. Untested alerting is broken alerting, so the result of
    every test call is kept and surfaced as an age on the dashboard."""

    __tablename__ = "self_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(24), default="call")
    contact_id: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "contact_id": self.contact_id,
            "outcome": self.outcome,
            "detail": self.detail,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
