"""Vehicle tracking history and ALPR search endpoints."""

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from schemas.surveillance import ALPRSearchResponse, TrackingHistoryResponse

router = APIRouter(prefix="/api/v1", tags=["surveillance"])


@router.get("/tracking/history", response_model=TrackingHistoryResponse)
async def get_tracking_history(
    plate_number: str | None = None,
    journey_id: UUID | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    camera_id: list[str] = Query(default=[]),
    location_id: list[UUID] = Query(default=[]),
    site_id: UUID | None = None,
    include_clips: bool = False,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Retrieve vehicle tracking history with locations and timestamps."""
    if not plate_number and not journey_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either plate_number or journey_id is required",
        )

    if not start:
        start = datetime.utcnow() - timedelta(days=1)
    if not end:
        end = datetime.utcnow()

    if (end - start).days > 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Time range cannot exceed 7 days",
        )

    # Placeholder - would query DB in production
    return TrackingHistoryResponse(
        plate_number=plate_number.upper() if plate_number else "",
        journey_id=journey_id,
        first_seen_at=start,
        last_seen_at=end,
        total_events=0,
        history=[],
        pagination={"page": page, "limit": limit, "total": 0},
    )


@router.get("/alpr/search", response_model=ALPRSearchResponse)
async def search_alpr(
    start: datetime,
    end: datetime,
    camera_id: list[str] = Query(default=[]),
    location_id: list[UUID] = Query(default=[]),
    site_id: UUID | None = None,
    plate_number: str | None = None,
    min_det_conf: float | None = None,
    min_ocr_conf: float | None = None,
    has_clip: bool | None = None,
    group_id: UUID | None = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Advanced search across ALPR detections."""
    if (end - start).days > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Time range cannot exceed 3 days for ad-hoc search",
        )

    return ALPRSearchResponse(
        total=0,
        results=[],
        pagination={"page": page, "limit": limit, "total": 0},
        filters_applied={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "camera_id": camera_id,
            "location_id": [str(loc) for loc in location_id],
        },
    )
