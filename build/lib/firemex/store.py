"""Persistence layer.

Synchronous SQLAlchemy behind ``asyncio.to_thread``. Deliberate: the query volume
is trivial (a handful of writes per incident), and sync sessions keep the code
readable and the driver choice boring.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import anyio
from sqlalchemy import create_engine, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from . import metrics
from .detect.base import Detection
from .incident.engine import Assessment, Incident
from .models import AlertRecord, Base, IncidentRecord, SelfTestRecord
from .notify.base import AlertResult
from .notify.dispatcher import DispatchRun

log = logging.getLogger(__name__)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Store:
    def __init__(self, database_url: str, echo: bool = False) -> None:
        if database_url.startswith("sqlite"):
            # SQLite is the zero-setup default for a single-box install; the
            # connection is shared across worker threads, hence the flag.
            path = database_url.split("///")[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                database_url, echo=echo, connect_args={"check_same_thread": False}
            )
        else:
            self.engine = create_engine(
                database_url, echo=echo, pool_pre_ping=True, pool_size=5, max_overflow=5
            )
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    # ---- incidents --------------------------------------------------------

    def _save_incident(
        self,
        incident: Incident,
        camera_name: str,
        location: str,
        detections: Sequence[Detection],
        assessment: Assessment | None,
        shadow_mode: bool,
    ) -> None:
        with self.session() as session:
            record = session.get(IncidentRecord, incident.id)
            if record is None:
                record = IncidentRecord(id=incident.id)
                session.add(record)
            record.camera_id = incident.camera_id
            record.camera_name = camera_name
            record.location = location
            record.severity = incident.severity.value
            record.labels = ",".join(sorted(incident.labels))
            record.peak_confidence = incident.peak_confidence
            record.growth_ratio = incident.growth_ratio
            record.frames_confirmed = incident.frames_confirmed
            record.envelope = (
                {
                    "x1": incident.envelope.x1,
                    "y1": incident.envelope.y1,
                    "x2": incident.envelope.x2,
                    "y2": incident.envelope.y2,
                }
                if incident.envelope
                else None
            )
            record.opened_at = dt.datetime.fromtimestamp(incident.opened_wall, dt.UTC)
            record.shadow_mode = shadow_mode
            record.detections = [d.as_dict() for d in detections]
            if assessment is not None:
                record.assessment = {
                    "hit_frames": assessment.hit_frames,
                    "frames_required": assessment.frames_required,
                    "cluster_size": assessment.cluster_size,
                    "stability_required": assessment.stability_required,
                    "growth_ratio": round(assessment.growth_ratio, 4),
                }
            session.commit()

    async def save_incident(
        self,
        incident: Incident,
        camera_name: str = "",
        location: str = "",
        detections: Sequence[Detection] = (),
        assessment: Assessment | None = None,
        shadow_mode: bool = False,
    ) -> None:
        await anyio.to_thread.run_sync(
            self._save_incident,
            incident,
            camera_name,
            location,
            list(detections),
            assessment,
            shadow_mode,
        )

    def _update_incident(self, incident_id: str, **fields) -> dict | None:
        with self.session() as session:
            record = session.get(IncidentRecord, incident_id)
            if record is None:
                return None
            for key, value in fields.items():
                setattr(record, key, value)
            session.commit()
            return record.as_dict()

    async def update_incident(self, incident_id: str, **fields) -> dict | None:
        return await anyio.to_thread.run_sync(
            lambda: self._update_incident(incident_id, **fields)
        )

    async def close_incident(self, incident: Incident) -> None:
        closed = incident.closed_wall or incident.opened_wall
        await self.update_incident(
            incident.id,
            closed_at=dt.datetime.fromtimestamp(closed, dt.UTC),
            severity=incident.severity.value,
            labels=",".join(sorted(incident.labels)),
            peak_confidence=incident.peak_confidence,
            growth_ratio=incident.growth_ratio,
        )

    async def set_evidence(
        self, incident_id: str, snapshot_path: str | None = None, clip_path: str | None = None
    ) -> None:
        fields = {}
        if snapshot_path:
            fields["snapshot_path"] = snapshot_path
        if clip_path:
            fields["clip_path"] = clip_path
        if fields:
            await self.update_incident(incident_id, **fields)

    def _record_run(self, run: DispatchRun) -> None:
        with self.session() as session:
            record = session.get(IncidentRecord, run.incident_id)
            if record is not None:
                record.alert_status = run.status.value
                if run.acknowledged_by:
                    record.acknowledged_by = run.acknowledged_by
                    record.acknowledged_at = record.acknowledged_at or _utcnow()
            existing = {
                (a.channel, a.contact_id, a.provider_id)
                for a in session.scalars(
                    select(AlertRecord).where(AlertRecord.incident_id == run.incident_id)
                )
            }
            for attempt in run.attempts:
                key = (attempt.channel, attempt.contact_id, attempt.provider_id)
                # Runs are published repeatedly as they progress; only append new rows.
                if key in existing and attempt.provider_id is not None:
                    continue
                existing.add(key)
                session.add(
                    AlertRecord(
                        incident_id=run.incident_id,
                        channel=attempt.channel,
                        contact_id=attempt.contact_id,
                        outcome=attempt.outcome.value,
                        provider_id=attempt.provider_id,
                        error=attempt.error,
                    )
                )
            session.commit()

    async def record_run(self, run: DispatchRun) -> None:
        await anyio.to_thread.run_sync(self._record_run, run)

    def _list_incidents(
        self,
        limit: int,
        offset: int,
        camera_id: str | None,
        review: str | None,
        unreviewed_only: bool,
    ) -> list[dict]:
        with self.session() as session:
            query = select(IncidentRecord).order_by(desc(IncidentRecord.opened_at))
            if camera_id:
                query = query.where(IncidentRecord.camera_id == camera_id)
            if review:
                query = query.where(IncidentRecord.review == review)
            if unreviewed_only:
                query = query.where(IncidentRecord.review.is_(None))
            records = session.scalars(query.limit(limit).offset(offset)).all()
            return [record.as_dict() for record in records]

    async def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: str | None = None,
        review: str | None = None,
        unreviewed_only: bool = False,
    ) -> list[dict]:
        return await anyio.to_thread.run_sync(
            self._list_incidents, limit, offset, camera_id, review, unreviewed_only
        )

    def _get_incident(self, incident_id: str) -> dict | None:
        with self.session() as session:
            record = session.get(IncidentRecord, incident_id)
            return record.as_dict() if record else None

    async def get_incident(self, incident_id: str) -> dict | None:
        return await anyio.to_thread.run_sync(self._get_incident, incident_id)

    def _incident_paths(self, incident_id: str) -> tuple[str | None, str | None]:
        with self.session() as session:
            record = session.get(IncidentRecord, incident_id)
            if record is None:
                return None, None
            return record.snapshot_path, record.clip_path

    async def incident_paths(self, incident_id: str) -> tuple[str | None, str | None]:
        return await anyio.to_thread.run_sync(self._incident_paths, incident_id)

    def _stats(self, since: dt.datetime) -> dict:
        with self.session() as session:
            total = session.scalar(
                select(func.count()).select_from(IncidentRecord).where(
                    IncidentRecord.opened_at >= since
                )
            )
            false_positives = session.scalar(
                select(func.count()).select_from(IncidentRecord).where(
                    IncidentRecord.opened_at >= since,
                    IncidentRecord.review == "false_positive",
                )
            )
            confirmed_real = session.scalar(
                select(func.count()).select_from(IncidentRecord).where(
                    IncidentRecord.opened_at >= since, IncidentRecord.review == "real"
                )
            )
            unreviewed = session.scalar(
                select(func.count()).select_from(IncidentRecord).where(
                    IncidentRecord.opened_at >= since, IncidentRecord.review.is_(None)
                )
            )
            last_test = session.scalars(
                select(SelfTestRecord).order_by(desc(SelfTestRecord.created_at)).limit(1)
            ).first()
            reviewed = (false_positives or 0) + (confirmed_real or 0)
            return {
                "window_start": since.isoformat(),
                "incidents": total or 0,
                "false_positives": false_positives or 0,
                "real": confirmed_real or 0,
                "unreviewed": unreviewed or 0,
                # Only meaningful once incidents have actually been reviewed, so it
                # is None rather than 0 when the queue is empty.
                "false_positive_rate": (
                    round((false_positives or 0) / reviewed, 4) if reviewed else None
                ),
                "last_self_test": last_test.as_dict() if last_test else None,
            }

    async def stats(self, days: int = 7) -> dict:
        since = _utcnow() - dt.timedelta(days=days)
        return await anyio.to_thread.run_sync(self._stats, since)

    def _record_self_test(
        self, kind: str, contact_id: str, outcome: str, detail: str | None
    ) -> None:
        with self.session() as session:
            session.add(
                SelfTestRecord(kind=kind, contact_id=contact_id, outcome=outcome, detail=detail)
            )
            session.commit()
        metrics.LAST_SELF_TEST.set(time.time())

    async def record_self_test(
        self, kind: str, contact_id: str, outcome: str, detail: str | None = None
    ) -> None:
        await anyio.to_thread.run_sync(self._record_self_test, kind, contact_id, outcome, detail)

    async def record_alert_result(self, incident_id: str, result: AlertResult) -> None:
        def _write() -> None:
            with self.session() as session:
                session.add(
                    AlertRecord(
                        incident_id=incident_id,
                        channel=result.channel,
                        contact_id=result.contact_id,
                        outcome=result.outcome.value,
                        provider_id=result.provider_id,
                        error=result.error,
                    )
                )
                session.commit()

        await anyio.to_thread.run_sync(_write)

    def dispose(self) -> None:
        self.engine.dispose()
