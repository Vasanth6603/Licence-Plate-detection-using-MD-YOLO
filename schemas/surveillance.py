"""API schemas for surveillance system."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LocationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    location_id: UUID
    site_name: str
    city: str | None = None
    state: str | None = None
    country: str | None = None
    lat: float | None = None
    lng: float | None = None


class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    camera_id: str
    location_id: UUID
    site_id: UUID | None = None
    name: str | None = None
    direction: str | None = None
    is_active: bool = True
    group_id: UUID | None = None


class TrackingHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    event_id: UUID
    detected_at: datetime
    camera_id: str
    location: LocationResponse
    plate_number: str
    bbox: list[int] | None = None
    det_conf: float | None = None
    ocr_conf: float | None = None
    angle_used: float | None = None
    track_id: str | None = None
    clip_id: UUID | None = None
    clip_url: str | None = None


class TrackingHistoryResponse(BaseModel):
    success: bool = True
    plate_number: str
    journey_id: UUID | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    total_events: int = 0
    history: list[TrackingHistoryItem] = []
    pagination: dict = {}


class ALPRSearchResponse(BaseModel):
    success: bool = True
    total: int = 0
    results: list[TrackingHistoryItem] = []
    pagination: dict = {}
    filters_applied: dict = {}
