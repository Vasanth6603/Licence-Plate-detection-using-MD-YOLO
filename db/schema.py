"""Database schema definitions for ALPR surveillance system."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

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
)
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Location(Base):
    __tablename__ = "locations"

    location_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    site_name: Mapped[str] = mapped_column(String(128), nullable=False)
    city: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    cameras: Mapped[list["Camera"]] = relationship(back_populates="location")


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    location_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("locations.location_id")
    )
    site_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    name: Mapped[str | None] = mapped_column(String(128))
    rtsp_url: Mapped[str | None] = mapped_column(Text)
    make_model: Mapped[str | None] = mapped_column(String(64))
    direction: Mapped[str | None] = mapped_column(String(32))
    roi_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    fps_limit: Mapped[int] = mapped_column(Integer, default=12)
    group_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    location: Mapped[Location] = relationship(back_populates="cameras")


class ALPRDetection(Base):
    __tablename__ = "alpr_detections"

    event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    plate_number: Mapped[str] = mapped_column(String(16), nullable=False)
    plate_normalized: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(32), ForeignKey("cameras.camera_id"))
    location_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    site_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    track_id: Mapped[str | None] = mapped_column(String(64))
    vehicle_journey_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    bbox_x1: Mapped[int | None] = mapped_column(Integer)
    bbox_y1: Mapped[int | None] = mapped_column(Integer)
    bbox_x2: Mapped[int | None] = mapped_column(Integer)
    bbox_y2: Mapped[int | None] = mapped_column(Integer)
    det_conf: Mapped[float | None] = mapped_column(Float)
    ocr_conf: Mapped[float | None] = mapped_column(Float)
    angle_used: Mapped[float | None] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(Text)
    clip_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    frame_id: Mapped[int | None] = mapped_column(Integer)
    source_ip: Mapped[str | None] = mapped_column(INET)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    __table_args__ = (
        Index("idx_alpr_plate_time", "plate_normalized", "detected_at"),
        Index("idx_alpr_cam_time", "camera_id", "detected_at"),
        Index("idx_alpr_loc_time", "location_id", "detected_at"),
        Index("idx_alpr_journey", "vehicle_journey_id", "detected_at"),
    )


class VehicleJourney(Base):
    __tablename__ = "vehicle_journeys"

    journey_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    plate_normalized: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_camera_id: Mapped[str | None] = mapped_column(String(32))
    last_camera_id: Mapped[str | None] = mapped_column(String(32))
    first_location_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    last_location_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    path_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class VideoClip(Base):
    __tablename__ = "video_clips"

    clip_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[str] = mapped_column(String(32), ForeignKey("cameras.camera_id"))
    location_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    storage_bucket: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="active")
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (Index("idx_clips_expire", "expires_at"),)
