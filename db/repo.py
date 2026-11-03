"""Repository layer for ALPR data."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import ALPRDetection, Camera, Location


class ALPRRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_detection(self, data: dict) -> ALPRDetection:
        det = ALPRDetection(**data)
        self.session.add(det)
        await self.session.commit()
        await self.session.refresh(det)
        return det

    async def get_tracking_history(
        self,
        plate_normalized: str | None = None,
        journey_id: UUID | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        camera_ids: list[str] | None = None,
        location_ids: list[UUID] | None = None,
        site_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        stmt = (
            select(ALPRDetection, Camera, Location)
            .join(Camera, Camera.camera_id == ALPRDetection.camera_id)
            .join(Location, Location.location_id == ALPRDetection.location_id)
        )
        filters = []
        if plate_normalized:
            filters.append(ALPRDetection.plate_normalized == plate_normalized.upper())
        if journey_id:
            filters.append(ALPRDetection.vehicle_journey_id == journey_id)
        if start:
            filters.append(ALPRDetection.detected_at >= start)
        if end:
            filters.append(ALPRDetection.detected_at <= end)
        if camera_ids:
            filters.append(ALPRDetection.camera_id.in_(camera_ids))
        if location_ids:
            filters.append(ALPRDetection.location_id.in_(location_ids))
        if site_id:
            filters.append(ALPRDetection.site_id == site_id)
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(ALPRDetection.detected_at.asc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return result.all()

    async def search_detections(self, start: datetime, end: datetime, **kwargs):
        stmt = (
            select(ALPRDetection, Camera, Location)
            .join(Camera, Camera.camera_id == ALPRDetection.camera_id)
            .join(Location, Location.location_id == ALPRDetection.location_id)
        )
        filters = [ALPRDetection.detected_at >= start, ALPRDetection.detected_at <= end]
        if kwargs.get("camera_ids"):
            filters.append(ALPRDetection.camera_id.in_(kwargs["camera_ids"]))
        if kwargs.get("location_ids"):
            filters.append(ALPRDetection.location_id.in_(kwargs["location_ids"]))
        if kwargs.get("site_id"):
            filters.append(ALPRDetection.site_id == kwargs["site_id"])
        if kwargs.get("plate_number"):
            filters.append(
                ALPRDetection.plate_normalized.like(f"%{kwargs['plate_number'].upper()}%")
            )
        if kwargs.get("min_det_conf"):
            filters.append(ALPRDetection.det_conf >= kwargs["min_det_conf"])
        if kwargs.get("min_ocr_conf"):
            filters.append(ALPRDetection.ocr_conf >= kwargs["min_ocr_conf"])
        stmt = (
            stmt.where(and_(*filters))
            .order_by(desc(ALPRDetection.detected_at))
            .limit(kwargs.get("limit", 100))
            .offset(kwargs.get("offset", 0))
        )
        result = await self.session.execute(stmt)
        return result.all()

    async def get_count(self, start: datetime, end: datetime, **kwargs):
        stmt = select(func.count(ALPRDetection.event_id))
        filters = [ALPRDetection.detected_at >= start, ALPRDetection.detected_at <= end]
        if kwargs.get("plate_normalized"):
            filters.append(ALPRDetection.plate_normalized == kwargs["plate_normalized"].upper())
        if filters:
            stmt = stmt.where(and_(*filters))
        result = await self.session.execute(stmt)
        return result.scalar() or 0
